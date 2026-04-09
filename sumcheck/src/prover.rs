//! O(n) Sumcheck prover implementation.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_field::Field;

use crate::virtual_poly::VirtualPolynomial;

/// Sumcheck prover state
pub struct SumcheckProver {}

impl SumcheckProver {
    /// Generates a Sumcheck proof of the sum of the virtual polynomial over the hypercube
    ///
    /// The proof consists of a sequence of univariate polynomials, one per var.
    pub fn prove<F: Field, Challenger: FieldChallenger<F>>(
        virtual_poly: &mut impl VirtualPolynomial<F>,
        challenger: &mut Challenger,
    ) -> Vec<Vec<F>> {
        let num_vars = virtual_poly.num_vars();
        let degree = virtual_poly.degree();
        let mut round_polys = Vec::with_capacity(num_vars);

        for _ in 0..num_vars {
            let current_vars = virtual_poly.num_vars();
            let half_size = 1 << (current_vars - 1);
            let mut evals = vec![F::ZERO; degree + 1];

            // Eval univar polynomial (at X = 0,1,...,n) by summing over remaining hypercube in O(n) space
            for (x_val, eval) in evals.iter_mut().enumerate() {
                let x = F::from_usize(x_val);
                let mut sum = F::ZERO;
                for i in 0..half_size {
                    sum += virtual_poly.eval_at(x, i);
                }
                *eval = sum; // Write sum directly into evals memory slice
            }

            // Observe the evaluations in the transcript to produce the next challenge
            for &eval in &evals {
                challenger.observe_algebra_element(eval);
            }

            let r = challenger.sample_algebra_element();

            // Fold the base polynomials in-place!
            virtual_poly.bind(r);

            round_polys.push(evals);
        }

        round_polys
    }
}
