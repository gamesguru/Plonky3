use std::time::Instant;

use p3_baby_bear::BabyBear;
use p3_code::{CodeOrFamily, IdentityCode};
use p3_field::PrimeCharacteristicRing;
use p3_keccak::KeccakF;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use p3_tensor_pcs::{MultilinearPcs, TensorPcs};
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

fn init_tracing() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    let _ = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .try_init();
}

type F = BabyBear;
const VECTOR_LEN: usize = p3_keccak::VECTOR_LEN;
type MyHash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
type MyCompress = CompressionFunctionFromHasher<MyHash, 2, 4>;
type MyMmcs =
    MerkleTreeMmcs<[F; VECTOR_LEN], [u64; VECTOR_LEN], SerializingHasher<MyHash>, MyCompress, 2, 4>;

#[test]
fn test_tensor_pcs_scaling_suite() {
    init_tracing();

    let hash = MyHash::new(KeccakF {});
    let compress = MyCompress::new(hash);
    let mmcs = MyMmcs::new(SerializingHasher::new(hash), compress, 0);

    println!("\n=== Tensor PCS Scaling Suite (All Directions) ===");

    // 1. Breadth Scaling (Fixed Depth, Varying Number of Polys)
    println!("\n--- Breadth Scaling (log_n=12, n=4096, varying m) ---");
    for m in [1, 10, 50, 100] {
        let n = 1 << 12;
        let code = IdentityCode { len: n };
        let pcs = TensorPcs::new(code, mmcs.clone());
        let evals = (0..m)
            .map(|_| RowMajorMatrix::new(vec![F::ZERO; n], 1))
            .collect::<Vec<_>>();

        let t0 = Instant::now();
        let _ = <TensorPcs<F, IdentityCode, MyMmcs> as MultilinearPcs<F, F>>::commit(&pcs, evals);
        let dur = t0.elapsed();
        println!(
            "  m = {:3} polynomials | Commit took {:?} ({:?} per poly)",
            m,
            dur,
            dur / m as u32
        );
    }

    // 2. Transposition Scaling (Depth)
    println!("\n--- Transposition Scaling (varying log_n) ---");
    for log_n in [10, 14, 18, 20] {
        let n = 1 << log_n;
        let width = 1 << (log_n / 2);
        let height = 1 << (log_n - log_n / 2);
        let mat_sq = RowMajorMatrix::new(vec![F::ZERO; n], width);

        let t0 = Instant::now();
        let mut _dummy = vec![F::ZERO; n];
        for r in 0..height {
            for c in 0..width {
                _dummy[c * height + r] = mat_sq.values[r * width + c];
            }
        }
        let dur = t0.elapsed();
        println!(
            "  log_n = {:2} (n = {:7}) | Transpose took {:?} ({:?} per element)",
            log_n,
            n,
            dur,
            dur / n as u32
        );
    }

    // 3. Folding Round Simulation (Linearity in Opening)
    println!("\n--- Folding Round Simulation (Sumcheck Step) ---");
    for log_n in [10, 14, 18, 20] {
        let n = 1 << log_n;
        let vals: Vec<F> = (0..n).map(|i| F::from_u32(i as u32)).collect();

        let t0 = Instant::now();
        let challenge = F::from_u32(12345);
        let mut _folded = Vec::with_capacity(n / 2);
        for i in 0..(n / 2) {
            _folded.push(vals[2 * i] + challenge * vals[2 * i + 1]);
        }
        let dur = t0.elapsed();
        println!(
            "  log_n = {:2} (n = {:7}) | Folding Round took {:?} ({:?} per input element)",
            log_n,
            n,
            dur,
            dur / n as u32
        );
    }

    // 4. Hadamard Product Scaling (Constraint Evaluation)
    println!("\n--- Hadamard Product Scaling (A * B) ---");
    for log_n in [10, 14, 18, 20] {
        let n = 1 << log_n;
        let vals_a: Vec<F> = (0..n).map(|i| F::from_u32(i as u32)).collect();
        let vals_b: Vec<F> = (0..n).map(|i| F::from_u32(i as u32 + 7)).collect();

        let t0 = Instant::now();
        let mut _res = Vec::with_capacity(n);
        for i in 0..n {
            _res.push(vals_a[i] * vals_b[i]);
        }
        let dur = t0.elapsed();
        println!(
            "  log_n = {:2} (n = {:7}) | Hadamard took {:?} ({:?} per element)",
            log_n,
            n,
            dur,
            dur / n as u32
        );
    }
}

#[test]
fn test_tensor_pcs_linearity_scaling() {
    init_tracing();

    let hash = MyHash::new(KeccakF {});
    let compress = MyCompress::new(hash);
    let mmcs = MyMmcs::new(SerializingHasher::new(hash), compress, 0);

    println!("\n--- Linearity Scaling (Depth up to 20 bits) ---");
    for log_n in [10, 14, 18, 20] {
        let n = 1 << log_n;
        let code = IdentityCode { len: n };
        let pcs = TensorPcs::new(code, mmcs.clone());

        let mat_a = RowMajorMatrix::new((0..n).map(|i| F::from_u32(i as u32)).collect(), 1);
        let mat_b = RowMajorMatrix::new((0..n).map(|i| F::from_u32(i as u32 + 1000)).collect(), 1);

        let t0 = Instant::now();
        let mat_sum = RowMajorMatrix::new(
            mat_a
                .values
                .iter()
                .zip(mat_b.values.iter())
                .map(|(a, b)| *a + *b)
                .collect(),
            1,
        );
        let encoded_sum = pcs.code.encode_batch(mat_sum);

        let enc_a = pcs.code.encode_batch(mat_a);
        let enc_b = pcs.code.encode_batch(mat_b);
        let enc_sum_expected = RowMajorMatrix::new(
            enc_a
                .values
                .iter()
                .zip(enc_b.values.iter())
                .map(|(a, b)| *a + *b)
                .collect(),
            1,
        );

        assert_eq!(encoded_sum, enc_sum_expected);
        let dur = t0.elapsed();
        println!(
            "  log_n = {:2} (n = {:7}) | Linearity Check took {:?} ({:?} per row)",
            log_n,
            n,
            dur,
            dur / n as u32
        );
    }
}
