use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_brakedown::BrakedownCode;
use p3_brakedown::sparse::CsrMatrix;
use p3_challenger::SerializingChallenger32;
use p3_code::IdentityCode;
use p3_field::PrimeCharacteristicRing;
use p3_keccak::{Keccak256Hash, KeccakF, VECTOR_LEN};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use p3_tensor_pcs::{StarkMultilinearPcs, TensorPcs};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type F = BabyBear;
type MyHash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
type MyCompress = CompressionFunctionFromHasher<MyHash, 2, 4>;
type MyMmcs =
    MerkleTreeMmcs<[F; VECTOR_LEN], [u64; VECTOR_LEN], SerializingHasher<MyHash>, MyCompress, 2, 4>;

fn setup_pcs(log_n: usize) -> TensorPcs<F, BrakedownCode<F, IdentityCode>, MyMmcs> {
    let log_r = log_n / 2;
    let height = 1 << log_r;

    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let a = CsrMatrix::<F>::rand_fixed_col_weight(&mut rng, height, height, 4);
    let b = CsrMatrix::<F>::rand_fixed_col_weight(&mut rng, height, height, 4);
    let inner_code = Box::new(IdentityCode { len: height });
    let code = BrakedownCode { a, b, inner_code };

    let hash = MyHash::new(KeccakF {});
    let compress = MyCompress::new(hash);
    let serial_hasher = SerializingHasher::new(hash);
    let mmcs = MyMmcs::new(serial_hasher, compress, 0);

    TensorPcs::new(code, mmcs, 40)
}

fn bench_tensor_pcs(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_pcs");
    group.sample_size(10);

    for &log_n in &[12, 14, 16] {
        let label = format!("log_n_{}", log_n);
        let pcs = setup_pcs(log_n);

        let n = 1 << log_n;
        let evals = vec![RowMajorMatrix::new(vec![F::ZERO; n], 1)];

        // Benchmark Commit
        group.bench_function(BenchmarkId::new("commit", &label), |b| {
            b.iter_batched(
                || evals.clone(),
                |evals_clone| {
                    let _commitment = black_box(<TensorPcs<F, _, _> as StarkMultilinearPcs<
                        F,
                        F,
                    >>::commit(&pcs, evals_clone));
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // Setup data for open / verify
        let (commitment, prover_data) =
            <TensorPcs<F, _, _> as StarkMultilinearPcs<F, F>>::commit(&pcs, evals.clone());
        let point = vec![F::ZERO; log_n];

        // Benchmark Open
        group.bench_function(BenchmarkId::new("open", &label), |b| {
            b.iter(|| {
                let mut challenger = SerializingChallenger32::from_hasher(vec![], Keccak256Hash);
                p3_challenger::CanObserve::observe(&mut challenger, commitment.clone());
                let _proof = black_box(<TensorPcs<F, _, _> as StarkMultilinearPcs<F, F>>::open(
                    &pcs,
                    &prover_data,
                    &point,
                    &mut challenger,
                ));
            });
        });

        let mut challenger = SerializingChallenger32::from_hasher(vec![], Keccak256Hash);
        p3_challenger::CanObserve::observe(&mut challenger, commitment.clone());
        let (opened_values, proof) = <TensorPcs<F, _, _> as StarkMultilinearPcs<F, F>>::open(
            &pcs,
            &prover_data,
            &point,
            &mut challenger,
        );

        // Benchmark Verify
        group.bench_function(BenchmarkId::new("verify", &label), |b| {
            b.iter(|| {
                let mut challenger = SerializingChallenger32::from_hasher(vec![], Keccak256Hash);
                p3_challenger::CanObserve::observe(&mut challenger, commitment.clone());
                let _res = black_box(<TensorPcs<F, _, _> as StarkMultilinearPcs<F, F>>::verify(
                    &pcs,
                    &commitment,
                    &point,
                    &opened_values,
                    &proof,
                    &mut challenger,
                ));
            });
        });
    }
}

criterion_group!(benches, bench_tensor_pcs);
criterion_main!(benches);
