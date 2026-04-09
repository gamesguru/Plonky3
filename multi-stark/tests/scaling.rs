use std::time::Instant;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::BabyBear;
use p3_challenger::SerializingChallenger32;
use p3_code::IdentityCode;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_tensor_pcs::TensorPcs;

struct DummyAir {
    width: usize,
}

impl<F> BaseAir<F> for DummyAir {
    fn width(&self) -> usize {
        self.width
    }
}

impl<AB: AirBuilder> Air<AB> for DummyAir
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let next = main.next_slice();
        // Formulated as = 0: next[0] - local[0] - 1
        builder
            .when_transition()
            .assert_zero(next[0] - local[0] - AB::Expr::ONE);
    }
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
fn test_multi_stark_scaling_suite() {
    println!("\n=== Multi-STARK Prover O(n) Scaling Suite ===");
    run_scaling_suite(&[10, 12, 14, 16]);
}

#[test]
#[ignore]
fn test_multi_stark_scaling_suite_heavy() {
    println!("\n=== Multi-STARK Prover O(n) Scaling Suite (Heavy) ===");
    run_scaling_suite(&[18, 20]);
}

fn run_scaling_suite(cases: &[usize]) {
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    let hash = Keccak256Hash;
    let compress = CompressionFunctionFromHasher::new(hash);
    let serial_hasher = SerializingHasher::new(hash);
    let mmcs = MerkleTreeMmcs::<F, u8, _, _, 2, 32>::new(serial_hasher, compress, 0);

    for &log_n in cases {
        let n = 1 << log_n;
        let width = 1; // TensorPCS takes width = 2^log_c
        let code = IdentityCode { len: n };
        let pcs = TensorPcs::new(code, mmcs.clone(), 40);

        let mut trace = RowMajorMatrix::new(vec![F::ZERO; n * width], width);
        for row in 0..n {
            trace.values[row * width] = F::from_usize(row);
        }

        let air = DummyAir { width };
        let mut prover_challenger =
            SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);

        let t0 = Instant::now();
        let proof = p3_multi_stark::prover::prove::<F, EF, _, _, _, _>(
            &pcs,
            &air,
            &mut prover_challenger,
            trace,
        )
        .unwrap();
        let dur = t0.elapsed();

        let label = format!("log_n = {:2} (n = 2^{})", log_n, log_n);
        let (d_val, d_unit) = split_dur(dur);
        let (p_val, p_unit) = split_dur(dur / n as u32);

        println!(
            "  {:22} | {:>9}: {:>8.2}{:<2} ({:>8.2}{:<2} per trace row)",
            label, "Prove", d_val, d_unit, p_val, p_unit
        );

        let mut verifier_challenger =
            SerializingChallenger32::<F, _>::from_hasher(vec![], Keccak256Hash);
        let t1 = Instant::now();
        p3_multi_stark::verifier::verify::<F, EF, _, _, _, _>(
            &pcs,
            &air,
            &mut verifier_challenger,
            &proof,
        )
        .unwrap();
        let v_dur = t1.elapsed();
        let (v_val, v_unit) = split_dur(v_dur);

        println!("  {:22} | {:>9}: {:>8.2}{:<2}", "", "Verify", v_val, v_unit);
    }
}
