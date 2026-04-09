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
            let a0 = self.a.as_slice()[2 * index];
            let a1 = self.a.as_slice()[2 * index + 1];
            let a_x = a0 + x * (a1 - a0);

            let b0 = self.b.as_slice()[2 * index];
            let b1 = self.b.as_slice()[2 * index + 1];
            let b_x = b0 + x * (b1 - b0);

            let c0 = self.c.as_slice()[2 * index];
            let c1 = self.c.as_slice()[2 * index + 1];
            let c_x = c0 + x * (c1 - c0);

            a_x * b_x - c_x
        }

        fn bind(&mut self, challenge: F) {
            self.a.fix_hi_var_mut(challenge);
            self.b.fix_hi_var_mut(challenge);
            self.c.fix_hi_var_mut(challenge);
        }
    }

    #[test]
    fn test_sumcheck_protocol() {
        type F = BabyBear;

        // Variables: v = 3 (8 evals per polynomial).
        // Polynomials: A, B, C.
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

        // Change 1st element of C.
        // Constraint: A * B - C
        // 1*2 - 0 = 2. (Other evals ARE zero, but not the 1st)
        let c_evals = vec![
            F::ZERO, // Provoke constraint error due to non-zero sum
            F::from_usize(6),
            F::from_usize(3),
            F::from_usize(20),
            F::from_usize(20),
            F::from_usize(48),
            F::from_usize(42),
            F::from_usize(56),
        ];

        // Expected sum: 2
        // (1*2-0) + 0 + 0 + 0 + 0 + 0 + 0 + 0 = 2
        let expected_sum = F::from_usize(2);

        // Clone for verifier use
        let a_poly = Poly::new(a_evals.clone());
        let b_poly = Poly::new(b_evals.clone());
        let c_poly = Poly::new(c_evals.clone());

        let mut virtual_poly = SimpleConstraint::new(a_evals, b_evals, c_evals);

        let mut prover_challenger =
            SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);

        let mut verifier_challenger =
            SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);

        let num_vars = virtual_poly.num_vars();
        let degree = virtual_poly.degree();

        let (proof, _z) = SumcheckProver::prove::<F, F, _>(
            expected_sum,
            &mut virtual_poly,
            &mut prover_challenger,
        );

        let (challenges, expected_eval) = SumcheckVerifier::verify::<F, F, _>(
            expected_sum,
            num_vars,
            degree,
            &proof,
            &mut verifier_challenger,
        )
        .expect("Sumcheck verification failed");

        let mut challenges_rev = challenges;
        challenges_rev.reverse();
        let challenges_point = p3_multilinear_util::point::Point::new(challenges_rev);
        let eval_a = a_poly.eval_ext(&challenges_point);
        let eval_b = b_poly.eval_ext(&challenges_point);
        let eval_c = c_poly.eval_ext(&challenges_point);

        assert_eq!(
            expected_eval,
            eval_a * eval_b - eval_c,
            "Verifier expected_eval did not match polynomial eval"
        );
    }

    #[derive(Debug)]
    struct HighDegreeConstraint<F: Field> {
        pub a: Poly<F>,
        pub b: Poly<F>,
        pub c: Poly<F>,
        pub d: Poly<F>,
    }

    impl<F: Field> VirtualPolynomial<F> for HighDegreeConstraint<F> {
        fn num_vars(&self) -> usize {
            self.a.num_vars()
        }

        fn degree(&self) -> usize {
            3 // A(x) * B(x) * C(x) - D(x) has degree 3
        }

        fn eval_at(&self, x: F, index: usize) -> F {
            let interp = |poly: &Poly<F>| {
                let p0 = poly.as_slice()[2 * index];
                let p1 = poly.as_slice()[2 * index + 1];
                p0 + x * (p1 - p0)
            };
            interp(&self.a) * interp(&self.b) * interp(&self.c) - interp(&self.d)
        }

        fn bind(&mut self, challenge: F) {
            self.a.fix_hi_var_mut(challenge);
            self.b.fix_hi_var_mut(challenge);
            self.c.fix_hi_var_mut(challenge);
            self.d.fix_hi_var_mut(challenge);
        }
    }

    #[test]
    fn test_high_degree_sumcheck_protocol() {
        type F = BabyBear;

        let n = 8; // v = 3
        let a_evals: Vec<F> = (0..n).map(|i| F::from_usize((i * 3) % 17)).collect();
        let b_evals: Vec<F> = (0..n).map(|i| F::from_usize((i * 5) % 19)).collect();
        let c_evals: Vec<F> = (0..n).map(|i| F::from_usize((i * 7) % 23)).collect();
        let d_evals: Vec<F> = (0..n).map(|i| F::from_usize((i * 11) % 29)).collect();

        let expected_sum: F = (0..n)
            .map(|i| a_evals[i] * b_evals[i] * c_evals[i] - d_evals[i])
            .sum();

        let a_poly = Poly::new(a_evals);
        let b_poly = Poly::new(b_evals);
        let c_poly = Poly::new(c_evals);
        let d_poly = Poly::new(d_evals);

        let mut virtual_poly = HighDegreeConstraint {
            a: a_poly.clone(),
            b: b_poly.clone(),
            c: c_poly.clone(),
            d: d_poly.clone(),
        };

        let mut prover_challenger =
            SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);

        let mut verifier_challenger =
            SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);

        let num_vars = virtual_poly.num_vars();
        let degree = virtual_poly.degree();

        let (proof, _z) = SumcheckProver::prove::<F, F, _>(
            expected_sum,
            &mut virtual_poly,
            &mut prover_challenger,
        );

        let (challenges, expected_eval) = SumcheckVerifier::verify::<F, F, _>(
            expected_sum,
            num_vars,
            degree,
            &proof,
            &mut verifier_challenger,
        )
        .expect("Degree 3 Sumcheck verification failed");

        let mut challenges_rev = challenges;
        challenges_rev.reverse();
        let challenges_point = p3_multilinear_util::point::Point::new(challenges_rev);
        let eval_a = a_poly.eval_ext(&challenges_point);
        let eval_b = b_poly.eval_ext(&challenges_point);
        let eval_c = c_poly.eval_ext(&challenges_point);
        let eval_d = d_poly.eval_ext(&challenges_point);

        assert_eq!(
            expected_eval,
            eval_a * eval_b * eval_c - eval_d,
            "Verifier expected_eval did not match constraint eval for degree 3"
        );
    }

    #[test]
    fn test_sumcheck_malicious_claim() {
        type F = BabyBear;

        let a_evals = vec![F::from_usize(1), F::from_usize(2)];
        let b_evals = vec![F::from_usize(2), F::from_usize(3)];
        let c_evals = vec![F::ZERO, F::from_usize(6)]; // 1*2-0 = 2, 2*3-6 = 0, sum = 2

        let mut virtual_poly = SimpleConstraint::new(a_evals, b_evals, c_evals);

        let mut prover_challenger =
            SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);
        let mut verifier_challenger =
            SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);

        let num_vars = virtual_poly.num_vars();
        let degree = virtual_poly.degree();

        let honest_sum = F::from_usize(2);
        let (proof, _z) =
            SumcheckProver::prove::<F, F, _>(honest_sum, &mut virtual_poly, &mut prover_challenger);

        // Byzantine prover claims sum is 0 (but really it's 2)
        let malicious_claim = F::ZERO;

        let verify_result = SumcheckVerifier::verify::<F, F, _>(
            malicious_claim,
            num_vars,
            degree,
            &proof,
            &mut verifier_challenger,
        );

        // Check that verifier catches & rejects lie
        assert!(
            verify_result.is_err(),
            "Verifier FAILED to reject malicious sumcheck claim!"
        );
    }

    #[derive(Debug)]
    struct BadVirtualPoly<F: Field> {
        vars: usize,
        _phantom: core::marker::PhantomData<F>,
    }

    impl<F: Field> VirtualPolynomial<F> for BadVirtualPoly<F> {
        fn num_vars(&self) -> usize {
            self.vars
        }

        fn degree(&self) -> usize {
            1
        }

        fn eval_at(&self, _x: F, _index: usize) -> F {
            F::ZERO
        }

        fn bind(&mut self, _challenge: F) {
            self.vars = 0; // Drops to 0 immediately!
        }
    }

    #[test]
    #[should_panic(expected = "Sumcheck needs virtual polynomial deg >= 1")]
    fn test_prover_degree_zero_panic() {
        type F = BabyBear;

        let _virtual_poly = BadVirtualPoly::<F> {
            vars: 1,
            _phantom: core::marker::PhantomData,
        };
        // Need to override degree for this test; define mock BadVirtualPoly, degree zero
        #[derive(Debug)]
        struct DegreeZeroPoly<F: Field> {
            _phantom: core::marker::PhantomData<F>,
        }
        impl<F: Field> VirtualPolynomial<F> for DegreeZeroPoly<F> {
            fn num_vars(&self) -> usize {
                1
            }
            fn degree(&self) -> usize {
                0
            }
            fn eval_at(&self, _x: F, _index: usize) -> F {
                F::ZERO
            }
            fn bind(&mut self, _challenge: F) {}
        }

        let mut poly = DegreeZeroPoly::<F> {
            _phantom: core::marker::PhantomData,
        };
        let mut prover_challenger =
            SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);

        let _ = SumcheckProver::prove::<F, F, _>(F::ZERO, &mut poly, &mut prover_challenger);
    }

    #[test]
    #[should_panic(expected = "Sumcheck requires num_vars >= 1")]
    fn test_prover_num_vars_zero_initial_panic() {
        type F = BabyBear;

        let mut virtual_poly = BadVirtualPoly::<F> {
            vars: 0,
            _phantom: core::marker::PhantomData,
        };

        let mut prover_challenger =
            SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);

        let _ =
            SumcheckProver::prove::<F, F, _>(F::ZERO, &mut virtual_poly, &mut prover_challenger);
    }

    #[test]
    #[should_panic(expected = "virtual polynomial num_vars not >= 1 during proof rounds")]
    fn test_prover_num_vars_zero_panic() {
        type F = BabyBear;

        let mut virtual_poly = BadVirtualPoly::<F> {
            vars: 2,
            _phantom: core::marker::PhantomData,
        };

        let mut prover_challenger =
            SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);

        // This will start the prove loop for 2 rounds.
        // Round 0: binds and drops vars to 0.
        // Round 1: checks current_vars >= 1 (and panics)
        let _ =
            SumcheckProver::prove::<F, F, _>(F::ZERO, &mut virtual_poly, &mut prover_challenger);
    }

    #[test]
    fn test_verifier_invalid_rounds() {
        type F = BabyBear;
        let mut challenger = SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);
        let claim = F::ZERO;
        let num_vars = 3;
        let degree = 2;
        let proof = vec![vec![F::ZERO; 3]; 2]; // Only 2 rounds instead of 3

        let result =
            SumcheckVerifier::verify::<F, F, _>(claim, num_vars, degree, &proof, &mut challenger);
        assert_eq!(
            result.err(),
            Some("Invalid number of rounds in sumcheck proof")
        );
    }

    #[test]
    fn test_verifier_invalid_degree() {
        type F = BabyBear;
        let mut challenger = SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);
        let claim = F::ZERO;
        let num_vars = 2;
        let degree = 2;
        let proof = vec![
            vec![F::ZERO; 3],
            vec![F::ZERO; 2], // Only 2 evals, but degree+1 (so, 3 expected)
        ];

        let result =
            SumcheckVerifier::verify::<F, F, _>(claim, num_vars, degree, &proof, &mut challenger);
        assert_eq!(
            result.err(),
            Some("Invalid univariate polynomial degree in sumcheck round")
        );
    }

    #[test]
    fn test_verifier_zero_vars() {
        type F = BabyBear;
        let mut challenger = SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);
        let result = SumcheckVerifier::verify::<F, F, _>(F::ZERO, 0, 1, &[], &mut challenger);
        assert_eq!(result.err(), Some("Sumcheck requires num_vars >= 1"));
    }

    #[test]
    fn test_verifier_zero_degree() {
        type F = BabyBear;
        let mut challenger = SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);
        let result =
            SumcheckVerifier::verify::<F, F, _>(F::ZERO, 1, 0, &[vec![F::ZERO]], &mut challenger);
        assert_eq!(result.err(), Some("Sumcheck requires degree >= 1"));
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

    fn run_sumcheck_scaling_suite(scales: &[usize]) {
        type F = BabyBear;

        println!("\n=== Sumcheck O(n) Scaling Suite ===");

        for &log_n in scales {
            let n = 1 << log_n;

            // Init pseudo-random eval arrays over hypercube
            let a_evals: Vec<F> = (0..n).map(|i| F::from_usize((i % 100) as usize)).collect();
            let b_evals: Vec<F> = (0..n)
                .map(|i| F::from_usize(((i + 7) % 100) as usize))
                .collect();
            let c_evals: Vec<F> = (0..n)
                .map(|i| F::from_usize(((i + 14) % 100) as usize))
                .collect();

            let mut virtual_poly =
                SimpleConstraint::new(a_evals.clone(), b_evals.clone(), c_evals.clone());

            // Sum claim over hypercube
            let claim: F = a_evals
                .into_iter()
                .zip(b_evals)
                .zip(c_evals)
                .map(|((a, b), c)| a * b - c)
                .sum();

            let mut prover_challenger =
                SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);

            let t0 = std::time::Instant::now();
            let (_proof, _z) =
                SumcheckProver::prove::<F, F, _>(claim, &mut virtual_poly, &mut prover_challenger);
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

    #[test]
    fn test_sumcheck_scaling_suite_light() {
        run_sumcheck_scaling_suite(&[10, 12, 14, 16, 18]);
    }

    #[test]
    #[ignore]
    fn test_sumcheck_scaling_suite_heavy() {
        run_sumcheck_scaling_suite(&[20, 22]);
    }
}
