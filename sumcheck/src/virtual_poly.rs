//! Virtual-polynomial frontend for the generic-degree sumcheck driver.
//!
//! This module is an adapter layer: it lets callers describe a polynomial by
//! lazy per-round evaluation over the native Boolean backend, while reusing the
//! existing [`crate::generic_degree::RoundProver`] transcript and proof format.

use alloc::vec::Vec;

use p3_field::Field;

use crate::generic_degree::RoundProver;

/// Lazily evaluated polynomial state for generic-degree sumcheck.
///
/// The active variable order matches [`RoundProver`]: each round binds the
/// current prefix variable, and `index` ranges over the remaining Boolean
/// hypercube after that variable is fixed to `x`.
pub trait VirtualPolynomial<EF: Field> {
    /// Number of variables still active in this prover state.
    fn num_vars(&self) -> usize;

    /// Per-variable degree of the current round polynomial.
    fn degree(&self) -> usize;

    /// Evaluate the current polynomial with the next variable set to `x`.
    ///
    /// `index` addresses the residual `(num_vars() - 1)`-variable Boolean
    /// hypercube in lexicographic order.
    fn eval_at(&self, x: EF, index: usize) -> EF;

    /// Bind the next variable to `challenge`, mutating the state for the next round.
    fn bind(&mut self, challenge: EF);
}

/// Adapter from [`VirtualPolynomial`] to the existing generic-degree sumcheck driver.
pub struct VirtualPolynomialRoundProver<VP> {
    polynomial: VP,
}

impl<VP> VirtualPolynomialRoundProver<VP> {
    /// Wrap a virtual polynomial so it can be used as a [`RoundProver`].
    #[inline]
    pub const fn new(polynomial: VP) -> Self {
        Self { polynomial }
    }

    /// Return the wrapped virtual polynomial state.
    #[inline]
    pub fn into_inner(self) -> VP {
        self.polynomial
    }
}

impl<EF, VP> RoundProver<EF> for VirtualPolynomialRoundProver<VP>
where
    EF: Field,
    VP: VirtualPolynomial<EF>,
{
    fn fold(&mut self, r: EF) {
        self.polynomial.bind(r);
    }

    fn round_poly(&self) -> Vec<EF> {
        let num_vars = self.polynomial.num_vars();
        assert!(
            num_vars > 0,
            "virtual-polynomial sumcheck: no active variables"
        );

        let degree = self.polynomial.degree();
        assert!(
            degree > 0,
            "virtual-polynomial sumcheck: degree must be > 0"
        );

        let residual_size = 1_usize
            .checked_shl((num_vars - 1) as u32)
            .expect("virtual-polynomial sumcheck: residual hypercube size overflow");

        let mut evals = Vec::with_capacity(degree);
        evals.push(sum_over_residual_hypercube(
            &self.polynomial,
            EF::ZERO,
            residual_size,
        ));
        evals.extend((2..=degree).map(|node| {
            sum_over_residual_hypercube(&self.polynomial, EF::from_usize(node), residual_size)
        }));
        evals
    }
}

fn sum_over_residual_hypercube<EF, VP>(polynomial: &VP, x: EF, residual_size: usize) -> EF
where
    EF: Field,
    VP: VirtualPolynomial<EF>,
{
    (0..residual_size)
        .map(|index| polynomial.eval_at(x, index))
        .sum()
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{DuplexChallenger, FieldChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::poly::Poly;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    fn fresh_challenger() -> Ch {
        let mut rng = SmallRng::seed_from_u64(0xDEADBEEF);
        let perm = Perm::new_from_rng_128(&mut rng);
        Ch::new(perm)
    }

    #[derive(Clone)]
    struct ProductPolynomial {
        factors: Vec<Poly<EF>>,
    }

    impl ProductPolynomial {
        fn new(factors: Vec<Poly<EF>>) -> Self {
            assert!(!factors.is_empty(), "at least one factor is required");
            let num_evals = factors[0].num_evals();
            assert!(
                factors.iter().all(|factor| factor.num_evals() == num_evals),
                "all factors must share the same Boolean hypercube"
            );
            Self { factors }
        }

        fn final_product(&self) -> EF {
            self.factors
                .iter()
                .map(|factor| factor.as_slice()[0])
                .product()
        }
    }

    impl VirtualPolynomial<EF> for ProductPolynomial {
        fn num_vars(&self) -> usize {
            self.factors[0].num_variables()
        }

        fn degree(&self) -> usize {
            self.factors.len()
        }

        fn eval_at(&self, x: EF, index: usize) -> EF {
            let half = self.factors[0].num_evals() / 2;
            self.factors
                .iter()
                .map(|factor| {
                    let lo = factor.as_slice()[index];
                    let hi = factor.as_slice()[index + half];
                    lo + (hi - lo) * x
                })
                .product()
        }

        fn bind(&mut self, challenge: EF) {
            for factor in &mut self.factors {
                factor.fix_prefix_var_mut(challenge);
            }
        }
    }

    struct NativeProductRoundProver {
        factors: Vec<Poly<EF>>,
    }

    impl RoundProver<EF> for NativeProductRoundProver {
        fn fold(&mut self, r: EF) {
            for factor in &mut self.factors {
                factor.fix_prefix_var_mut(r);
            }
        }

        fn round_poly(&self) -> Vec<EF> {
            let degree = self.factors.len();
            let residual_size = 1 << (self.factors[0].num_variables() - 1);
            let mut evals = Vec::with_capacity(degree);
            evals.push(native_sum_over_residual_hypercube(
                &self.factors,
                EF::ZERO,
                residual_size,
            ));
            evals.extend((2..=degree).map(|node| {
                native_sum_over_residual_hypercube(
                    &self.factors,
                    EF::from_usize(node),
                    residual_size,
                )
            }));
            evals
        }
    }

    fn native_sum_over_residual_hypercube(
        factors: &[Poly<EF>],
        node: EF,
        residual_size: usize,
    ) -> EF {
        let half = factors[0].num_evals() / 2;
        (0..residual_size)
            .map(|index| {
                factors
                    .iter()
                    .map(|factor| {
                        let lo = factor.as_slice()[index];
                        let hi = factor.as_slice()[index + half];
                        lo + (hi - lo) * node
                    })
                    .product::<EF>()
            })
            .sum()
    }

    fn random_product_instance(log_m: usize) -> (ProductPolynomial, EF) {
        let mut rng = SmallRng::seed_from_u64(123);
        let m = 1 << log_m;
        let columns: Vec<Vec<F>> = (0..3)
            .map(|_| (0..m).map(|_| rng.random()).collect())
            .collect();

        let claimed_sum = (0..m)
            .map(|i| {
                columns
                    .iter()
                    .map(|column| EF::from(column[i]))
                    .product::<EF>()
            })
            .sum();
        let factors = columns
            .into_iter()
            .map(|column| Poly::new(column.into_iter().map(EF::from).collect()))
            .collect();
        (ProductPolynomial::new(factors), claimed_sum)
    }

    #[test]
    fn virtual_polynomial_adapter_matches_native_round_prover_transcript() {
        let log_m = 6;
        let (virtual_product, claimed_sum) = random_product_instance(log_m);

        let mut native = NativeProductRoundProver {
            factors: virtual_product.factors.clone(),
        };
        let mut virtual_prover = VirtualPolynomialRoundProver::new(virtual_product);

        let mut native_challenger = fresh_challenger();
        native_challenger.observe_algebra_element(claimed_sum);
        let (native_proof, native_challenges) =
            native.prove::<F, _>(&mut native_challenger, log_m, 3, 0, claimed_sum);

        let mut virtual_challenger = fresh_challenger();
        virtual_challenger.observe_algebra_element(claimed_sum);
        let (virtual_proof, virtual_challenges) =
            virtual_prover.prove::<F, _>(&mut virtual_challenger, log_m, 3, 0, claimed_sum);

        assert_eq!(virtual_proof.claimed_sum, native_proof.claimed_sum);
        assert_eq!(virtual_proof.round_polys, native_proof.round_polys);
        assert_eq!(virtual_challenges, native_challenges);

        let mut verifier_challenger = fresh_challenger();
        let (verifier_challenges, final_sum) = virtual_proof
            .verify(&mut verifier_challenger, log_m, 3, 0)
            .unwrap();
        assert_eq!(verifier_challenges, virtual_challenges);
        assert_eq!(final_sum, virtual_prover.into_inner().final_product());
    }
}
