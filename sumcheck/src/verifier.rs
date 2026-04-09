//! Sumcheck verifier implementation

use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field, batch_multiplicative_inverse};

pub struct SumcheckVerifier;

impl SumcheckVerifier {
    pub fn verify<F, EF, Challenger>(
        claim: EF,
        num_vars: usize,
        degree: usize,
        proof: &[Vec<EF>],
        challenger: &mut Challenger,
    ) -> Result<(Vec<EF>, EF), &'static str>
    where
        F: Field,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    {
        if num_vars == 0 {
            return Err("Sumcheck requires num_vars >= 1");
        }
        if degree == 0 {
            return Err("Sumcheck requires degree >= 1");
        }
        if proof.len() != num_vars {
            return Err("Invalid number of rounds in sumcheck proof");
        }

        // Bind the Fiat–Shamir transcript to the public statement before
        // observing prover messages, mirroring the prover's transcript order.
        challenger.observe_algebra_element(claim);
        challenger.observe_algebra_element(EF::from_usize(num_vars));
        challenger.observe_algebra_element(EF::from_usize(degree));
        let mut expected_sum = claim;
        let mut challenges = Vec::with_capacity(num_vars);

        // TODO: pre-calculate fractions at compile time (save CPU cycles)

        // Pre-compute denoms, independent of point `r`
        let mut denominators = Vec::with_capacity(degree + 1);
        for i in 0..=degree {
            let mut den = EF::ONE;
            let x_i = EF::from_usize(i);
            for j in 0..=degree {
                if i != j {
                    den *= x_i - EF::from_usize(j);
                }
            }
            if den.is_zero() {
                return Err("Zero denom during inverse eval; degree may exceed field capacity");
            }
            denominators.push(den);
        }

        // "Batch invert" all denoms synchronously; O(d) inversions -> O(1) inversions
        let inv_denominators = batch_multiplicative_inverse(&denominators);

        // TODO: hardcode formulas for simple polynomials (d=2, d=3), no non-native/dynamic math

        for round_evals in proof {
            if round_evals.len() != degree + 1 || degree == 0 {
                return Err("Invalid univariate polynomial degree in sumcheck round");
            }

            // Check P(0) + P(1) == expected_sum
            let p_0 = round_evals[0];
            let p_1 = round_evals[1];
            if p_0 + p_1 != expected_sum {
                return Err("Sumcheck round polynomial sum breaks from expected value");
            }

            // Observe the polynomial in the transcript
            for &eval in round_evals {
                challenger.observe_algebra_element(eval);
            }

            // Sample challenge
            let r = challenger.sample_algebra_element();
            challenges.push(r);

            // Interpolate & eval P(r) to get expected sum for next round
            expected_sum = Self::lagrange_evaluate(round_evals, r, &inv_denominators)?;
        }

        Ok((challenges, expected_sum))
    }

    /// Eval degree-d univar polynomial GIVEN its evals over integer domain: `0..=degree`.
    fn lagrange_evaluate<F: Field>(
        evals: &[F],
        point: F,
        inv_denominators: &[F],
    ) -> Result<F, &'static str> {
        let d = evals.len() - 1;

        let mut result = F::ZERO;
        for (i, &y_val) in evals.iter().enumerate() {
            let mut num = F::ONE;

            for j in 0..=d {
                if i != j {
                    let x_j = F::from_usize(j);
                    num *= point - x_j;
                }
            }

            result += y_val * num * inv_denominators[i];
        }

        Ok(result)
    }
}
