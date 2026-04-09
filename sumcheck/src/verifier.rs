//! Sumcheck verifier implementation

use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_field::Field;

/// Sumcheck verifier state
pub struct SumcheckVerifier {}

impl SumcheckVerifier {
    /// Verifies a sumcheck proof.
    /// Returns random eval point & expected eval of polynomial at that point.
    pub fn verify<F: Field, Challenger: FieldChallenger<F>>(
        claim: F,
        num_vars: usize,
        degree: usize,
        proof: &[Vec<F>],
        challenger: &mut Challenger,
    ) -> Result<(Vec<F>, F), &'static str> {
        if proof.len() != num_vars {
            return Err("Invalid number of rounds in sumcheck proof");
        }

        let mut expected_sum = claim;
        let mut challenges = Vec::with_capacity(num_vars);

        for round_evals in proof {
            if round_evals.len() != degree + 1 {
                return Err("Invalid univariate polynomial degree in sumcheck round");
            }

            // Check P(0) + P(1) == expected_sum
            let p_0 = round_evals[0];
            let p_1 = round_evals[1];
            if p_0 + p_1 != expected_sum {
                return Err("Sumcheck round polynomial does not sum to expected value");
            }

            // Observe the polynomial in the transcript
            for &eval in round_evals {
                challenger.observe_algebra_element(eval);
            }

            // Sample challenge
            let r = challenger.sample_algebra_element();
            challenges.push(r);

            // Interpolate & eval P(r) to get expected sum for next round
            expected_sum = Self::lagrange_evaluate(round_evals, r);
        }

        Ok((challenges, expected_sum))
    }

    /// Evaluates a n-polynomial GIVEN its evals over boolean domain: `0, 1, ..., n`.
    fn lagrange_evaluate<F: Field>(evals: &[F], point: F) -> F {
        let mut result = F::ZERO;
        let d = evals.len() - 1;

        for (i, &y_val) in evals.iter().enumerate() {
            let x_i = F::from_usize(i);
            let mut num = F::ONE;
            let mut den = F::ONE;

            for j in 0..=d {
                if i != j {
                    let x_j = F::from_usize(j);
                    num *= point - x_j;
                    den *= x_i - x_j;
                }
            }
            result += y_val * (num * den.inverse());
        }

        result
    }
}
