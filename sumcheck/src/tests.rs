#[cfg(test)]
mod test {
    extern crate std;
    use alloc::vec::Vec;
    use alloc::{format, vec};
    use std::println;

    use p3_baby_bear::BabyBear;
    use p3_challenger::SerializingChallenger32;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_keccak::Keccak256Hash;
    use p3_multilinear_util::poly::Poly;

    use crate::prover::SumcheckProver;
    use crate::verifier::SumcheckVerifier;
    use crate::virtual_poly::VirtualPolynomial;

    #[derive(Debug)]
    struct SimpleConstraint<F: Field> {
        pub a: Poly<F>,
        pub b: Poly<F>,
        pub c: Poly<F>,
    }

    impl<F: Field> SimpleConstraint<F> {
        pub fn new(a: Vec<F>, b: Vec<F>, c: Vec<F>) -> Self {
            Self {
                a: Poly::new(a),
                b: Poly::new(b),
                c: Poly::new(c),
            }
        }
    }

    impl<F: Field> VirtualPolynomial<F> for SimpleConstraint<F> {
        fn num_vars(&self) -> usize {
            self.a.num_vars()
        }

        fn degree(&self) -> usize {
            2 // A(x) * B(x) - C(x) is degree 2
        }

        fn eval_at(&self, x: F, index: usize) -> F {
            let half = 1 << (self.num_vars() - 1);

            let a0 = self.a.as_slice()[index];
            let a1 = self.a.as_slice()[index + half];
            let a_x = a0 + x * (a1 - a0);

            let b0 = self.b.as_slice()[index];
            let b1 = self.b.as_slice()[index + half];
            let b_x = b0 + x * (b1 - b0);

            let c0 = self.c.as_slice()[index];
            let c1 = self.c.as_slice()[index + half];
            let c_x = c0 + x * (c1 - c0);

            a_x * b_x - c_x
        }

        fn bind(&mut self, challenge: F) {
            self.a.fix_lo_var_mut(challenge);
            self.b.fix_lo_var_mut(challenge);
            self.c.fix_lo_var_mut(challenge);
        }
    }

    #[test]
    fn test_sumcheck_protocol() {
        type F = BabyBear;

        // Variables: v = 3 (8 evals per polynomial)
        // Init some polynomials A, B, C.
        let a_evals = vec![
            F::from_usize(1),
            F::from_usize(2),
            F::from_usize(3),
            F::from_usize(4),
            F::from_usize(5),
            F::from_usize(6),
            F::from_usize(7),
            F::from_usize(8),
        ];

        let b_evals = vec![
            F::from_usize(2),
            F::from_usize(3),
            F::from_usize(1),
            F::from_usize(5),
            F::from_usize(4),
            F::from_usize(8),
            F::from_usize(6),
            F::from_usize(7),
        ];

        let c_evals = vec![
            F::from_usize(2),
            F::from_usize(6),
            F::from_usize(3),
            F::from_usize(20),
            F::from_usize(20),
            F::from_usize(48),
            F::from_usize(42),
            F::from_usize(56),
        ];

        // The constraint is A * B - C.
        // Sum exactly evaluates to:
        // (1*2-2) + (2*3-6) + (3*1-3) + (4*5-20) + (5*4-20) + (6*8-48) + (7*6-42) + (8*7-56)
        // 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 = 0.
        let mut virtual_poly = SimpleConstraint::new(a_evals, b_evals, c_evals);

        let mut prover_challenger =
            SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);

        let mut verifier_challenger =
            SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);

        let num_vars = virtual_poly.num_vars();
        let degree = virtual_poly.degree();
        let expected_sum = F::ZERO;

        let proof = SumcheckProver::prove(&mut virtual_poly, &mut prover_challenger);

        let verify_result = SumcheckVerifier::verify(
            expected_sum,
            num_vars,
            degree,
            &proof,
            &mut verifier_challenger,
        );

        assert!(verify_result.is_ok(), "Sumcheck verification failed");
    }

    fn split_dur(d: std::time::Duration) -> (f64, &'static str) {
        let ns = d.as_nanos() as f64;
        if ns < 1000.0 {
            (ns, "ns")
        } else if ns < 1_000_000.0 {
            (ns / 1000.0, "µs")
        } else if ns < 1_000_000_000.0 {
            (ns / 1_000_000.0, "ms")
        } else {
            (ns / 1_000_000_000.0, "s")
        }
    }

    #[test]
    fn test_sumcheck_scaling_suite() {
        type F = BabyBear;

        println!("\n=== Sumcheck O(n) Scaling Suite ===");

        for log_n in [10, 12, 14, 16, 18, 20, 22] {
            let n = 1 << log_n;

            // Initialize pseudo-random evaluation arrays over the boolean hypercube
            let a_evals: Vec<F> = (0..n).map(|i| F::from_usize((i % 100) as usize)).collect();
            let b_evals: Vec<F> = (0..n)
                .map(|i| F::from_usize(((i + 7) % 100) as usize))
                .collect();
            let c_evals: Vec<F> = (0..n)
                .map(|i| F::from_usize(((i + 14) % 100) as usize))
                .collect();

            let mut virtual_poly = SimpleConstraint::new(a_evals, b_evals, c_evals);

            let mut prover_challenger =
                SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);

            let t0 = std::time::Instant::now();
            let _proof = SumcheckProver::prove(&mut virtual_poly, &mut prover_challenger);
            let dur = t0.elapsed();

            let label = format!("log_n = {:2} (n = 2^{})", log_n, log_n);
            let (d_val, d_unit) = split_dur(dur);
            let (p_val, p_unit) = split_dur(dur / n as u32);

            println!(
                "  {:22} | {:>9}: {:>8.2}{:<2} ({:>8.2}{:<2} per element)",
                label, "Prove", d_val, d_unit, p_val, p_unit
            );
        }
    }
}
