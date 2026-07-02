#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::time::Instant;

use p3_baby_bear::BabyBear;
use p3_code::{CodeOrFamily, IdentityCode};
use p3_field::PrimeCharacteristicRing;
use p3_keccak::{KeccakF, VECTOR_LEN};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use p3_tensor_pcs::{StarkMultilinearPcs, TensorPcs};

type F = BabyBear;
type MyHash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
type MyCompress = CompressionFunctionFromHasher<MyHash, 2, 4>;
type MyMmcs =
    MerkleTreeMmcs<[F; VECTOR_LEN], [u64; VECTOR_LEN], SerializingHasher<MyHash>, MyCompress, 2, 4>;

fn get_breadth_test_dims() -> &'static [&'static [usize]] {
    if cfg!(debug_assertions) {
        &[&[12, 1, 10], &[14, 1]]
    } else {
        &[
            &[12, 1, 10, 100],
            &[14, 1, 10, 100],
            &[16, 1, 10, 100],
            &[18, 1, 10, 100],
            &[20, 1, 10, 100],
            &[22, 1, 10, 100],
        ]
    }
}

fn get_scaling_bounds() -> &'static [usize] {
    if cfg!(debug_assertions) {
        &[10, 14, 18]
    } else {
        &[10, 14, 18, 20, 22, 24, 26, 28, 30]
    }
}

/// Asymptotic performance benchmark suite for Tensor PCS.
pub fn main() {
    let hash = MyHash::new(KeccakF {});
    let compress = MyCompress::new(hash);
    let mmcs = MyMmcs::new(SerializingHasher::new(hash), compress, 0);

    println!("\n=== Tensor PCS Scaling Suite (all directions) ===");

    benchmark_breadth_scaling(&mmcs);
    benchmark_transposition_scaling();
    benchmark_folding_round_simulation();
    benchmark_hadamard_product_scaling();
    benchmark_linearity_scaling(&mmcs);
}

fn benchmark_breadth_scaling(mmcs: &MyMmcs) {
    println!("\n--- Breadth Scaling (varying log_n and m) ---");

    for row in get_breadth_test_dims() {
        let log_n = row[0];
        let m_cases = &row[1..];
        let n = 1 << log_n;

        for &m in m_cases {
            let code = IdentityCode { len: n };
            let pcs = TensorPcs::new(code, mmcs.clone(), 40);
            let evals = (0..m)
                .map(|_| RowMajorMatrix::new(vec![F::ZERO; n], 1))
                .collect::<Vec<_>>();

            let t0 = Instant::now();
            let _ = <TensorPcs<F, IdentityCode, MyMmcs> as StarkMultilinearPcs<F, F>>::commit(
                &pcs, evals,
            );
            let dur = t0.elapsed();

            let label = format!("log_n = {log_n:2}, m = {m:3}");
            let (d_val, d_unit) = split_dur(dur);
            let (p_val, p_unit) = split_dur(dur / m as u32);
            let dim_str = format!("{m}x{n} = {} cells", format_count(m * n));
            println!(
                "  {:22} | {:>9}: {:>8.2}{:<2} ({:>8.2}{:<2} per poly) | Dim: {dim_str}",
                label, "Commit", d_val, d_unit, p_val, p_unit
            );
        }
    }
}

fn benchmark_transposition_scaling() {
    println!("\n--- Transposition scaling (varying log_n) ---");
    for &log_n in get_scaling_bounds() {
        let n = 1 << log_n;
        let width = 1 << (log_n / 2);
        let height = 1 << (log_n - log_n / 2);
        let mat_sq = RowMajorMatrix::new(vec![F::ZERO; n], width);

        let t0 = Instant::now();
        let mut dummy = vec![F::ZERO; n];

        // Optimize via cache-oblivious block transposition (Morton-order Z-curve traversal or block tiling).
        // Dividing the matrix into tiny 64x64 blocks that fit perfectly into the L1/L2 cache before
        // transposing them speeds up the prover's initial setup phase exponentially for massive graphs.
        let block_size = 64;
        for r_block in (0..height).step_by(block_size) {
            for c_block in (0..width).step_by(block_size) {
                let r_max = std::cmp::min(r_block + block_size, height);
                let c_max = std::cmp::min(c_block + block_size, width);
                for r in r_block..r_max {
                    for c in c_block..c_max {
                        dummy[c * height + r] = mat_sq.values[r * width + c];
                    }
                }
            }
        }
        let dur = t0.elapsed();

        let label = format!("log_n = {log_n:2} (n = 2^{log_n})");
        let (d_val, d_unit) = split_dur(dur);
        let (p_val, p_unit) = split_dur(dur / n as u32);
        let dim_str = format!("{height}x{width} = {} cells", format_count(n));
        println!(
            "  {:22} | {:>9}: {:>8.2}{:<2} ({:>8.2}{:<2} per element) | Dim: {dim_str}",
            label, "Transpose", d_val, d_unit, p_val, p_unit
        );
    }
}

fn benchmark_folding_round_simulation() {
    println!("\n--- Folding round simulation (Sumcheck Step) ---");
    for &log_n in get_scaling_bounds() {
        let n = 1 << log_n;
        let vals: Vec<F> = (0..n).map(|i| F::from_u32(i as u32)).collect();

        let t0 = Instant::now();
        let challenge = F::from_u32(12345);
        let mut folded = Vec::with_capacity(n / 2);
        for i in 0..(n / 2) {
            folded.push(vals[2 * i] + challenge * vals[2 * i + 1]);
        }
        let dur = t0.elapsed();

        let label = format!("log_n = {log_n:2} (n = 2^{log_n})");
        let (d_val, d_unit) = split_dur(dur);
        let (p_val, p_unit) = split_dur(dur / n as u32);
        let dim_str = format!("{n} -> {} elements", n / 2);
        println!(
            "  {:22} | {:>9}: {:>8.2}{:<2} ({:>8.2}{:<2} per input) | Dim: {dim_str}",
            label, "Folding", d_val, d_unit, p_val, p_unit
        );
    }
}

fn benchmark_hadamard_product_scaling() {
    println!("\n--- Hadamard Product Scaling (A * B) ---");
    for &log_n in get_scaling_bounds() {
        let n = 1 << log_n;
        let vals_a: Vec<F> = (0..n).map(|i| F::from_u32(i as u32)).collect();
        let vals_b: Vec<F> = (0..n).map(|i| F::from_u32(i as u32 + 7)).collect();

        let t0 = Instant::now();
        let mut res = Vec::with_capacity(n);
        for i in 0..n {
            res.push(vals_a[i] * vals_b[i]);
        }
        let dur = t0.elapsed();

        let label = format!("log_n = {log_n:2} (n = 2^{log_n})");
        let (d_val, d_unit) = split_dur(dur);
        let (p_val, p_unit) = split_dur(dur / n as u32);
        let dim_str = format!("{n} x {n} -> {n} cells");
        println!(
            "  {:22} | {:>9}: {:>8.2}{:<2} ({:>8.2}{:<2} per element) | Dim: {dim_str}",
            label, "Hadamard", d_val, d_unit, p_val, p_unit
        );
    }
}

fn benchmark_linearity_scaling(mmcs: &MyMmcs) {
    println!("\n--- Linearity scaling (depths up to 28 bits) ---");
    for &log_n in get_scaling_bounds() {
        // Cap to log_n <= 28 to prevent OOM kills on massive matrix allocations
        if log_n > 28 {
            continue;
        }
        let n = 1 << log_n;
        let code = IdentityCode { len: n };
        let pcs = TensorPcs::new(code, mmcs.clone(), 40);

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

        let label = format!("log_n = {log_n:2} (n = 2^{log_n})");
        let (d_val, d_unit) = split_dur(dur);
        let (p_val, p_unit) = split_dur(dur / n as u32);
        let dim_str = format!("{n}x1 = {n} cells");
        println!(
            "  {:22} | {:>9}: {:>8.2}{:<2} ({:>8.2}{:<2} per row) | Dim: {dim_str}",
            label, "Linearity", d_val, d_unit, p_val, p_unit
        );
    }
}

fn split_dur(d: std::time::Duration) -> (f64, &'static str) {
    let ns = d.as_secs_f64() * 1_000_000_000.0;
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

fn format_count(count: usize) -> String {
    if count < 1000 {
        format!("{count}")
    } else if count < 1_000_000 {
        let k = count / 1000;
        let rem = (count % 1000) / 100;
        if rem > 0 {
            format!("{k}.{rem}k")
        } else {
            format!("{k}k")
        }
    } else if count < 1_000_000_000 {
        let m = count / 1_000_000;
        let rem = (count % 1_000_000) / 100_000;
        if rem > 0 {
            format!("{m}.{rem}M")
        } else {
            format!("{m}M")
        }
    } else {
        let b = count / 1_000_000_000;
        let rem = (count % 1_000_000_000) / 100_000_000;
        if rem > 0 {
            format!("{b}.{rem}B")
        } else {
            format!("{b}B")
        }
    }
}
