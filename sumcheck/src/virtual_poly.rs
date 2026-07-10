//! Virtual-polynomial frontend for the generic-degree sumcheck driver.
//!
//! This module is an adapter layer: it lets callers describe a polynomial by
//! lazy per-round evaluation over the native Boolean backend, while reusing the
//! existing [`crate::generic_degree::RoundProver`] transcript and proof format.
//!
//! # Variable Order and Indexing Convention
//!
//! When implementing [`VirtualPolynomial`], the variables are bound prefix-first (high bit first).
//! This aligns with the variable folding order of [`p3_multilinear_util::poly::Poly::fix_prefix_var_mut`].
//!
//! In each round, the variable being bound is the prefix (i.e. the first active variable).
//! The parameter `index` ranges over the remaining Boolean hypercube of size `2^(num_vars - 1)`,
//! addressed in lexicographic order.
//!
//! Thus, `eval_at(x, index)` must evaluate the polynomial where:
//! - The next variable to be bound is set to `x`.
//! - The remaining variables are bound to the bits of `index`, where the first remaining
//!   variable corresponds to the most-significant bit of `index`, down to the last remaining
//!   variable corresponding to the least-significant bit.
//!
//! ## Worked Example
//!
//! Suppose a polynomial $P(x_0, x_1, x_2)$ has $num\_vars() = 3$ active variables at the start of
//! a round. The next variable to be bound is $x_0$.
//!
//! There are $num\_vars() - 1 = 2$ remaining variables, so `index` ranges from $0$ to $3$:
//! - `index = 0` (binary `00`) maps to remaining variables $(x_1, x_2) = (0, 0)$.
//! - `index = 1` (binary `01`) maps to remaining variables $(x_1, x_2) = (0, 1)$.
//! - `index = 2` (binary `10`) maps to remaining variables $(x_1, x_2) = (1, 0)$.
//! - `index = 3` (binary `11`) maps to remaining variables $(x_1, x_2) = (1, 1)$.
//!
//! Therefore, evaluating at a point `x`:
//! - `eval_at(x, 0)` returns $P(x, 0, 0)$.
//! - `eval_at(x, 1)` returns $P(x, 0, 1)$.
//! - `eval_at(x, 2)` returns $P(x, 1, 0)$.
//! - `eval_at(x, 3)` returns $P(x, 1, 1)$.
//!
//! In code, this matches the indexing of a multilinear polynomial split into two halves:
//! `lo` (where the prefix variable is $0$) and `hi` (where the prefix variable is $1$).
//! The evaluation is `lo + (hi - lo) * x`, where `lo` is located at `index` and `hi` is located
//! at `index + half`.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::Field;
use p3_multilinear_util::poly::Poly;

use crate::generic_degree::RoundProver;

/// Lazily evaluated polynomial state for generic-degree sumcheck.
///
/// The active variable order matches [`RoundProver`]: each round binds the
/// current prefix variable, and `index` ranges over the remaining Boolean
/// hypercube after that variable is fixed to `x`.
pub trait VirtualPolynomial<EF: Field> {
    /// Number of variables still active in this prover state.
    fn num_vars(&self) -> usize;

    /// Per-variable degree of the virtual polynomial.
    ///
    /// This must be constant across all rounds and must match the `degree` parameter
    /// passed to the sumcheck prover driver [`RoundProver::prove`] and the verifier.
    fn degree(&self) -> usize;

    /// Evaluate the current polynomial with the next variable set to `x`.
    ///
    /// `index` addresses the residual `(num_vars() - 1)`-variable Boolean
    /// hypercube in lexicographic order.
    fn eval_at(&self, x: EF, index: usize) -> EF;

    /// Evaluate the current polynomial at multiple points for a fixed residual index.
    ///
    /// The default implementation evaluates each point individually via [`Self::eval_at`].
    /// Implementing this directly can allow caching computations across points.
    #[inline]
    fn eval_at_nodes(&self, index: usize, nodes: &[EF], evals: &mut [EF]) {
        debug_assert_eq!(nodes.len(), evals.len());
        for (eval, &x) in evals.iter_mut().zip(nodes) {
            *eval = self.eval_at(x, index);
        }
    }

    /// Bind the next variable to `challenge`, mutating the state for the next round.
    fn bind(&mut self, challenge: EF);
}

/// A virtual polynomial representing the product of multiple multilinear polynomials.
///
/// This is the canonical representation of a product polynomial.
#[derive(Debug, Clone)]
pub struct ProductPolynomial<EF: Field> {
    factors: Vec<Poly<EF>>,
}

impl<EF: Field> ProductPolynomial<EF> {
    /// Create a new product polynomial from a list of multilinear factors.
    ///
    /// # Panics
    ///
    /// Panics if the list of factors is empty, or if not all factors have the
    /// same number of variables.
    pub fn new(factors: Vec<Poly<EF>>) -> Self {
        assert!(
            !factors.is_empty(),
            "virtual-polynomial sumcheck: at least one factor is required"
        );
        let num_vars = factors[0].num_variables();
        assert!(
            factors
                .iter()
                .all(|factor| factor.num_variables() == num_vars),
            "virtual-polynomial sumcheck: all factors must share the same number of variables"
        );
        Self { factors }
    }

    /// Return the factors of this product polynomial.
    pub fn factors(&self) -> &[Poly<EF>] {
        &self.factors
    }

    /// Compute the final product of the constant polynomial (evaluation at the final point).
    pub fn final_product(&self) -> EF {
        self.factors
            .iter()
            .map(|factor| factor.as_slice()[0])
            .product()
    }
}

impl<EF: Field> VirtualPolynomial<EF> for ProductPolynomial<EF> {
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

    fn eval_at_nodes(&self, index: usize, nodes: &[EF], evals: &mut [EF]) {
        debug_assert_eq!(nodes.len(), evals.len());
        let half = self.factors[0].num_evals() / 2;
        evals.fill(EF::ONE);
        for factor in self.factors.iter() {
            let lo = factor.as_slice()[index];
            let hi = factor.as_slice()[index + half];
            let diff = hi - lo;
            for (eval, &x) in evals.iter_mut().zip(nodes) {
                *eval *= lo + diff * x;
            }
        }
    }

    fn bind(&mut self, challenge: EF) {
        for factor in &mut self.factors {
            factor.fix_prefix_var_mut(challenge);
        }
    }
}

/// Adapter from [`VirtualPolynomial`] to the existing generic-degree sumcheck driver.
#[derive(Debug, Clone)]
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

    /// Get a reference to the inner virtual polynomial.
    #[inline]
    pub const fn inner(&self) -> &VP {
        &self.polynomial
    }

    /// Get a mutable reference to the inner virtual polynomial.
    #[inline]
    pub const fn inner_mut(&mut self) -> &mut VP {
        &mut self.polynomial
    }
}

impl<EF, VP> RoundProver<EF> for VirtualPolynomialRoundProver<VP>
where
    EF: Field,
    VP: VirtualPolynomial<EF> + Send + Sync,
{
    fn fold(&mut self, r: EF) {
        self.polynomial.bind(r);
    }

    fn round_poly(&self) -> Vec<EF> {
        let degree = self.polynomial.degree();
        assert!(
            degree > 0,
            "virtual-polynomial sumcheck: degree must be > 0"
        );

        let num_vars = self.polynomial.num_vars();
        assert!(
            num_vars > 0,
            "virtual-polynomial sumcheck: no active variables"
        );

        assert!(
            num_vars <= usize::BITS as usize,
            "virtual-polynomial sumcheck: num_vars exceeds word size"
        );
        let residual_vars = num_vars - 1;
        let residual_size = 1_usize << residual_vars;

        let mut nodes = Vec::with_capacity(degree);
        nodes.push(EF::ZERO);
        for i in 2..=degree {
            nodes.push(EF::from_usize(i));
        }

        use p3_maybe_rayon::prelude::*;
        (0..residual_size)
            .into_par_iter()
            .par_fold_reduce(
                || (vec![EF::ZERO; degree], vec![EF::ZERO; degree]),
                |(mut acc, mut evals_buf), index| {
                    self.polynomial.eval_at_nodes(index, &nodes, &mut evals_buf);
                    for (a, &e) in acc.iter_mut().zip(&evals_buf) {
                        *a += e;
                    }
                    (acc, evals_buf)
                },
                |(mut acc1, buf), (acc2, _)| {
                    for (a1, a2) in acc1.iter_mut().zip(acc2) {
                        *a1 += a2;
                    }
                    (acc1, buf)
                },
            )
            .0
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{DuplexChallenger, FieldChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
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

    fn random_product_instance(log_m: usize) -> (ProductPolynomial<EF>, EF) {
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
        for log_m in [1, 3, 6] {
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

    #[test]
    fn test_edge_cases() {
        // num_vars = 1 (single round)
        {
            let log_m = 1;
            let (virtual_product, claimed_sum) = random_product_instance(log_m);
            let mut virtual_prover = VirtualPolynomialRoundProver::new(virtual_product);
            let mut challenger = fresh_challenger();
            challenger.observe_algebra_element(claimed_sum);
            let (proof, _challenges) =
                virtual_prover.prove::<F, _>(&mut challenger, log_m, 3, 0, claimed_sum);
            assert_eq!(proof.round_polys.len(), 1);
        }

        // degree = 1 (single factor)
        {
            let log_m = 3;
            let mut rng = SmallRng::seed_from_u64(123);
            let m = 1 << log_m;
            let values: Vec<F> = (0..m).map(|_| rng.random()).collect();
            let claimed_sum = values.iter().map(|&x| EF::from(x)).sum();
            let poly = Poly::new(values.into_iter().map(EF::from).collect());
            let virtual_product = ProductPolynomial::new(vec![poly]);
            let mut virtual_prover = VirtualPolynomialRoundProver::new(virtual_product);
            let mut challenger = fresh_challenger();
            challenger.observe_algebra_element(claimed_sum);
            let (proof, _challenges) =
                virtual_prover.prove::<F, _>(&mut challenger, log_m, 1, 0, claimed_sum);
            assert_eq!(proof.round_polys.len(), 3);
        }
    }

    #[test]
    #[should_panic(expected = "virtual-polynomial sumcheck: no active variables")]
    fn test_zero_vars_panic() {
        let poly = Poly::new(vec![EF::ONE]);
        let virtual_product = ProductPolynomial::new(vec![poly]);
        let virtual_prover = VirtualPolynomialRoundProver::new(virtual_product);
        let _ = virtual_prover.round_poly();
    }

    #[test]
    #[should_panic(expected = "virtual-polynomial sumcheck: degree must be > 0")]
    fn test_zero_degree_panic() {
        let virtual_product: ProductPolynomial<EF> = ProductPolynomial {
            factors: Vec::new(),
        };
        let virtual_prover = VirtualPolynomialRoundProver::new(virtual_product);
        let _ = virtual_prover.round_poly();
    }

    #[test]
    fn test_nonzero_pow_bits() {
        let log_m = 3;
        let (virtual_product, claimed_sum) = random_product_instance(log_m);
        let mut virtual_prover = VirtualPolynomialRoundProver::new(virtual_product);
        let mut challenger = fresh_challenger();
        challenger.observe_algebra_element(claimed_sum);
        let (proof, _challenges) =
            virtual_prover.prove::<F, _>(&mut challenger, log_m, 3, 2, claimed_sum);
        assert_eq!(proof.pow_witnesses.len(), log_m);
    }

    #[derive(Clone)]
    struct CompositePolynomial {
        f: Poly<EF>,
        g: Poly<EF>,
        h: Poly<EF>,
    }

    impl VirtualPolynomial<EF> for CompositePolynomial {
        fn num_vars(&self) -> usize {
            self.f.num_variables()
        }

        fn degree(&self) -> usize {
            2
        }

        fn eval_at(&self, x: EF, index: usize) -> EF {
            let half = self.f.num_evals() / 2;
            let f_lo = self.f.as_slice()[index];
            let f_hi = self.f.as_slice()[index + half];
            let f_val = f_lo + (f_hi - f_lo) * x;

            let g_lo = self.g.as_slice()[index];
            let g_hi = self.g.as_slice()[index + half];
            let g_val = g_lo + (g_hi - g_lo) * x;

            let h_lo = self.h.as_slice()[index];
            let h_hi = self.h.as_slice()[index + half];
            let h_val = h_lo + (h_hi - h_lo) * x;

            f_val * g_val + h_val
        }

        fn eval_at_nodes(&self, index: usize, nodes: &[EF], evals: &mut [EF]) {
            debug_assert_eq!(nodes.len(), evals.len());
            let half = self.f.num_evals() / 2;
            let f_lo = self.f.as_slice()[index];
            let f_hi = self.f.as_slice()[index + half];
            let f_diff = f_hi - f_lo;

            let g_lo = self.g.as_slice()[index];
            let g_hi = self.g.as_slice()[index + half];
            let g_diff = g_hi - g_lo;

            let h_lo = self.h.as_slice()[index];
            let h_hi = self.h.as_slice()[index + half];
            let h_diff = h_hi - h_lo;

            for (eval, &x) in evals.iter_mut().zip(nodes) {
                let f_val = f_lo + f_diff * x;
                let g_val = g_lo + g_diff * x;
                let h_val = h_lo + h_diff * x;
                *eval = f_val * g_val + h_val;
            }
        }

        fn bind(&mut self, challenge: EF) {
            self.f.fix_prefix_var_mut(challenge);
            self.g.fix_prefix_var_mut(challenge);
            self.h.fix_prefix_var_mut(challenge);
        }
    }

    struct NativeCompositeRoundProver {
        f: Poly<EF>,
        g: Poly<EF>,
        h: Poly<EF>,
    }

    impl RoundProver<EF> for NativeCompositeRoundProver {
        fn fold(&mut self, r: EF) {
            self.f.fix_prefix_var_mut(r);
            self.g.fix_prefix_var_mut(r);
            self.h.fix_prefix_var_mut(r);
        }

        fn round_poly(&self) -> Vec<EF> {
            let degree = 2;
            let residual_size = 1 << (self.f.num_variables() - 1);
            let mut evals = Vec::with_capacity(degree);
            evals.push(native_composite_sum_over_residual_hypercube(
                &self.f,
                &self.g,
                &self.h,
                EF::ZERO,
                residual_size,
            ));
            evals.push(native_composite_sum_over_residual_hypercube(
                &self.f,
                &self.g,
                &self.h,
                EF::from_usize(2),
                residual_size,
            ));
            evals
        }
    }

    fn native_composite_sum_over_residual_hypercube(
        f: &Poly<EF>,
        g: &Poly<EF>,
        h: &Poly<EF>,
        node: EF,
        residual_size: usize,
    ) -> EF {
        let half = f.num_evals() / 2;
        (0..residual_size)
            .map(|index| {
                let f_lo = f.as_slice()[index];
                let f_hi = f.as_slice()[index + half];
                let f_val = f_lo + (f_hi - f_lo) * node;

                let g_lo = g.as_slice()[index];
                let g_hi = g.as_slice()[index + half];
                let g_val = g_lo + (g_hi - g_lo) * node;

                let h_lo = h.as_slice()[index];
                let h_hi = h.as_slice()[index + half];
                let h_val = h_lo + (h_hi - h_lo) * node;

                f_val * g_val + h_val
            })
            .sum()
    }

    #[test]
    fn test_non_product_composite_polynomial() {
        let log_m = 4;
        let mut rng = SmallRng::seed_from_u64(123);
        let m = 1 << log_m;

        let f_vals: Vec<F> = (0..m).map(|_| rng.random()).collect();
        let g_vals: Vec<F> = (0..m).map(|_| rng.random()).collect();
        let h_vals: Vec<F> = (0..m).map(|_| rng.random()).collect();

        let claimed_sum = (0..m)
            .map(|i| {
                let f_val = EF::from(f_vals[i]);
                let g_val = EF::from(g_vals[i]);
                let h_val = EF::from(h_vals[i]);
                f_val * g_val + h_val
            })
            .sum();

        let f = Poly::new(f_vals.into_iter().map(EF::from).collect());
        let g = Poly::new(g_vals.into_iter().map(EF::from).collect());
        let h = Poly::new(h_vals.into_iter().map(EF::from).collect());

        let composite = CompositePolynomial {
            f: f.clone(),
            g: g.clone(),
            h: h.clone(),
        };

        let mut native = NativeCompositeRoundProver { f, g, h };
        let mut virtual_prover = VirtualPolynomialRoundProver::new(composite);

        let mut native_challenger = fresh_challenger();
        native_challenger.observe_algebra_element(claimed_sum);
        let (native_proof, native_challenges) =
            native.prove::<F, _>(&mut native_challenger, log_m, 2, 0, claimed_sum);

        let mut virtual_challenger = fresh_challenger();
        virtual_challenger.observe_algebra_element(claimed_sum);
        let (virtual_proof, virtual_challenges) =
            virtual_prover.prove::<F, _>(&mut virtual_challenger, log_m, 2, 0, claimed_sum);

        assert_eq!(virtual_proof.claimed_sum, native_proof.claimed_sum);
        assert_eq!(virtual_proof.round_polys, native_proof.round_polys);
        assert_eq!(virtual_challenges, native_challenges);
    }
}
