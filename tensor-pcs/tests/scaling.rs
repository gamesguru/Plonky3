use std::time::Instant;

use p3_baby_bear::BabyBear;
use p3_code::{CodeOrFamily, IdentityCode};
use p3_field::PrimeCharacteristicRing;
use p3_keccak::KeccakF;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use p3_tensor_pcs::{MultilinearPcs, TensorPcs};

type F = BabyBear;
const VECTOR_LEN: usize = p3_keccak::VECTOR_LEN;
type MyHash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
type MyCompress = CompressionFunctionFromHasher<MyHash, 2, 4>;
type MyMmcs =
    MerkleTreeMmcs<[F; VECTOR_LEN], [u64; VECTOR_LEN], SerializingHasher<MyHash>, MyCompress, 2, 4>;

#[test]
fn test_tensor_pcs_scaling_suite() {
    let hash = MyHash::new(KeccakF {});
    let compress = MyCompress::new(hash);
    let mmcs = MyMmcs::new(SerializingHasher::new(hash), compress, 0);

    println!("\n=== Tensor PCS Scaling Suite (All Directions) ===");

    // Breadth Scaling (Varying Depth n and Number of Polys m)
    println!("\n--- Breadth Scaling (varying log_n and m) ---");
    for log_n in [12, 14, 16] {
        let n = 1 << log_n;
        // Only do large m=100 for the small depth log_n=12,14 to save time
        let m_cases = if log_n <= 14 {
            vec![1, 10, 100]
        } else {
            vec![1, 10]
        };
        for m in m_cases {
            let code = IdentityCode { len: n };
            let pcs = TensorPcs::new(code, mmcs.clone());
            let evals = (0..m)
                .map(|_| RowMajorMatrix::new(vec![F::ZERO; n], 1))
                .collect::<Vec<_>>();

            let t0 = Instant::now();
            let _ =
                <TensorPcs<F, IdentityCode, MyMmcs> as MultilinearPcs<F, F>>::commit(&pcs, evals);
            let dur = t0.elapsed();
            let label = format!("log_n = {:2}, m = {:3}", log_n, m);
            let (d_val, d_unit) = split_dur(dur);
            let (p_val, p_unit) = split_dur(dur / m as u32);
            println!(
                "  {:22} | {:>9}: {:>8.2}{:<2} ({:>8.2}{:<2} per poly)",
                label, "Commit", d_val, d_unit, p_val, p_unit
            );
        }
    }

    // Transposition Scaling (Depth)
    println!("\n--- Transposition Scaling (varying log_n) ---");
    for log_n in [10, 14, 18, 20, 22, 24] {
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
        let label = format!("log_n = {:2} (n = 2^{})", log_n, log_n);
        let (d_val, d_unit) = split_dur(dur);
        let (p_val, p_unit) = split_dur(dur / n as u32);
        println!(
            "  {:22} | {:>9}: {:>8.2}{:<2} ({:>8.2}{:<2} per element)",
            label, "Transpose", d_val, d_unit, p_val, p_unit
        );
    }

    // Folding Round Simulation (Linearity in Opening)
    println!("\n--- Folding Round Simulation (Sumcheck Step) ---");
    for log_n in [10, 14, 18, 20, 22, 24] {
        let n = 1 << log_n;
        let vals: Vec<F> = (0..n).map(|i| F::from_u32(i as u32)).collect();

        let t0 = Instant::now();
        let challenge = F::from_u32(12345);
        let mut _folded = Vec::with_capacity(n / 2);
        for i in 0..(n / 2) {
            _folded.push(vals[2 * i] + challenge * vals[2 * i + 1]);
        }
        let dur = t0.elapsed();
        let label = format!("log_n = {:2} (n = 2^{})", log_n, log_n);
        let (d_val, d_unit) = split_dur(dur);
        let (p_val, p_unit) = split_dur(dur / n as u32);
        println!(
            "  {:22} | {:>9}: {:>8.2}{:<2} ({:>8.2}{:<2} per input)",
            label, "Folding", d_val, d_unit, p_val, p_unit
        );
    }

    // Hadamard Product Scaling (Constraint Evaluation)
    println!("\n--- Hadamard Product Scaling (A * B) ---");
    for log_n in [10, 14, 18, 20, 22, 24] {
        let n = 1 << log_n;
        let vals_a: Vec<F> = (0..n).map(|i| F::from_u32(i as u32)).collect();
        let vals_b: Vec<F> = (0..n).map(|i| F::from_u32(i as u32 + 7)).collect();

        let t0 = Instant::now();
        let mut _res = Vec::with_capacity(n);
        for i in 0..n {
            _res.push(vals_a[i] * vals_b[i]);
        }
        let dur = t0.elapsed();
        let label = format!("log_n = {:2} (n = 2^{})", log_n, log_n);
        let (d_val, d_unit) = split_dur(dur);
        let (p_val, p_unit) = split_dur(dur / n as u32);
        println!(
            "  {:22} | {:>9}: {:>8.2}{:<2} ({:>8.2}{:<2} per element)",
            label, "Hadamard", d_val, d_unit, p_val, p_unit
        );
    }

    // Linearity Scaling (Depth)
    println!("\n--- Linearity Scaling (Depth up to 24 bits) ---");
    for log_n in [10, 14, 18, 20, 22, 24] {
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
        let label = format!("log_n = {:2} (n = 2^{})", log_n, log_n);
        let (d_val, d_unit) = split_dur(dur);
        let (p_val, p_unit) = split_dur(dur / n as u32);
        println!(
            "  {:22} | {:>9}: {:>8.2}{:<2} ({:>8.2}{:<2} per row)",
            label, "Linearity", d_val, d_unit, p_val, p_unit
        );
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
