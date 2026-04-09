use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;

use crate::virtual_poly::VirtualPolynomial;

#[cfg(feature = "parallel")]
trait VpBounds: core::marker::Sync + core::marker::Send {}
#[cfg(feature = "parallel")]
impl<T: ?Sized + core::marker::Sync + core::marker::Send> VpBounds for T {}

#[cfg(not(feature = "parallel"))]
trait VpBounds {}
#[cfg(not(feature = "parallel"))]
impl<T: ?Sized> VpBounds for T {}

pub struct SumcheckProver;

impl SumcheckProver {
    #[cfg(feature = "parallel")]
    pub fn prove<F, EF, Challenger>(
        claim: EF,
        virtual_poly: &mut (impl VirtualPolynomial<EF> + core::marker::Sync + core::marker::Send),
        challenger: &mut Challenger,
    ) -> (Vec<Vec<EF>>, Vec<EF>)
    where
        F: Field,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    {
        Self::prove_internal(claim, virtual_poly, challenger)
    }

    #[cfg(not(feature = "parallel"))]
    pub fn prove<F, EF, Challenger>(
        claim: EF,
        virtual_poly: &mut impl VirtualPolynomial<EF>,
        challenger: &mut Challenger,
    ) -> (Vec<Vec<EF>>, Vec<EF>)
    where
        F: Field,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    {
        Self::prove_internal(claim, virtual_poly, challenger)
    }

    // Bypass Sync for safe, serial fallback
    fn prove_internal<F, EF, Challenger, VP>(
        claim: EF,
        virtual_poly: &mut VP,
        challenger: &mut Challenger,
    ) -> (Vec<Vec<EF>>, Vec<EF>)
    where
        F: Field,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
        VP: VirtualPolynomial<EF> + ?Sized + VpBounds,
    {
        let num_vars = virtual_poly.num_vars();
        assert!(num_vars >= 1, "Sumcheck requires num_vars >= 1");
        let degree = virtual_poly.degree();
        assert!(degree >= 1, "Sumcheck needs virtual polynomial deg >= 1");

        // Bind Fiat–Shamir challenges.
        // Claim should be absorbed by caller/verifier path.
        challenger.observe_algebra_element(claim);
        challenger.observe_algebra_element(EF::from_usize(num_vars));
        challenger.observe_algebra_element(EF::from_usize(degree));

        let mut round_polys = Vec::with_capacity(num_vars);
        let mut z_challenges = Vec::with_capacity(num_vars);
        let x_domain: Vec<EF> = (0..=degree).map(|i| EF::from_usize(i)).collect();

        for _ in 0..num_vars {
            let current_vars = virtual_poly.num_vars();
            assert!(
                current_vars >= 1,
                "virtual polynomial num_vars must stay >= 1 during proof rounds"
            );
            let half_size = 1 << (current_vars - 1);
            let x_domain = &x_domain;
            let vp = &*virtual_poly;

            #[cfg(feature = "parallel")]
            let evals = (0..half_size).into_par_iter().par_fold_reduce(
                || {
                    // TODO: replace slow heap allocation with faster stack service
                    vec![EF::ZERO; degree + 1]
                },
                |mut acc, i| {
                    for (x_val, eval) in acc.iter_mut().enumerate() {
                        let x = x_domain[x_val];
                        *eval += vp.eval_at(x, i);
                    }
                    acc
                },
                |mut acc, local| {
                    for (a, b) in acc.iter_mut().zip(local) {
                        *a += b;
                    }
                    acc
                },
            );

            #[cfg(not(feature = "parallel"))]
            let evals = (0..half_size).fold(vec![EF::ZERO; degree + 1], |mut acc, i| {
                for (x_val, eval) in acc.iter_mut().enumerate() {
                    let x = x_domain[x_val];
                    *eval += vp.eval_at(x, i);
                }
                acc
            });

            for &eval in &evals {
                challenger.observe_algebra_element(eval);
            }

            let r: EF = challenger.sample_algebra_element();
            z_challenges.push(r); // Save Z
            virtual_poly.bind(r);

            round_polys.push(evals);
        }

        (round_polys, z_challenges)
    }
}
