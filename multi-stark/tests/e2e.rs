extern crate alloc;

use alloc::vec;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::BabyBear;
use p3_brakedown::BrakedownCode;
use p3_challenger::SerializingChallenger32;
use p3_code::IdentityCode;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_tensor_pcs::TensorPcs;
use rand::SeedableRng;

pub struct DummyAir;

impl<F> BaseAir<F> for DummyAir {
    fn width(&self) -> usize {
        1
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

        // Assert: next[0] = local[0] + 1
        // Formulated as = 0: next[0] - local[0] - 1
        builder
            .when_transition()
            .assert_zero(next[0] - local[0] - AB::Expr::ONE);
    }
}

#[test]
fn test_e2e_identity() {
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    let height = 1024;
    let width = 1;
    let mut trace_values = vec![F::ZERO; height * width];

    for i in 0..height {
        trace_values[i * width] = F::from_usize(i);
    }
    let trace = RowMajorMatrix::new(trace_values, width);

    let code = IdentityCode { len: height };
    let hash = Keccak256Hash;
    let compress = CompressionFunctionFromHasher::new(hash);
    let serial_hasher = SerializingHasher::new(hash);
    let mmcs = MerkleTreeMmcs::<F, u8, _, _, 2, 32>::new(serial_hasher, compress, 0);
    let tensor_pcs = TensorPcs::new(code, mmcs, 40);

    let air = DummyAir;

    let mut prover_challenger = SerializingChallenger32::<F, _>::from_hasher(vec![], hash);

    let proof = p3_multi_stark::prover::prove::<F, EF, _, _, _, _>(
        &tensor_pcs,
        &air,
        &mut prover_challenger,
        trace,
    )
    .expect("Prover failed");

    let mut verifier_challenger = SerializingChallenger32::<F, _>::from_hasher(vec![], hash);

    p3_multi_stark::verifier::verify::<F, EF, _, _, _, _>(
        &tensor_pcs,
        &air,
        &mut verifier_challenger,
        &proof,
    )
    .expect("Verifier failed");
}

#[test]
fn test_e2e_brakedown() {
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    let height = 1024;
    let width = 1;
    let mut trace_values = vec![F::ZERO; height * width];

    for i in 0..height {
        trace_values[i * width] = F::from_usize(i);
    }
    let trace = RowMajorMatrix::new(trace_values, width);

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let a =
        p3_brakedown::sparse::CsrMatrix::<F>::rand_fixed_col_weight(&mut rng, height, height, 4);
    let b =
        p3_brakedown::sparse::CsrMatrix::<F>::rand_fixed_col_weight(&mut rng, height, height, 4);
    let inner_code = alloc::boxed::Box::new(IdentityCode { len: height });
    let code = BrakedownCode { a, b, inner_code };

    let hash = Keccak256Hash;
    let compress = CompressionFunctionFromHasher::new(hash);
    let serial_hasher = SerializingHasher::new(hash);
    let mmcs = MerkleTreeMmcs::<F, u8, _, _, 2, 32>::new(serial_hasher, compress, 0);
    let tensor_pcs = TensorPcs::new(code, mmcs, 40);

    let air = DummyAir;

    let mut prover_challenger = SerializingChallenger32::<F, _>::from_hasher(vec![], hash);
    let proof = p3_multi_stark::prover::prove::<F, EF, _, _, _, _>(
        &tensor_pcs,
        &air,
        &mut prover_challenger,
        trace,
    )
    .expect("Prover failed");

    let mut verifier_challenger = SerializingChallenger32::<F, _>::from_hasher(vec![], hash);
    p3_multi_stark::verifier::verify::<F, EF, _, _, _, _>(
        &tensor_pcs,
        &air,
        &mut verifier_challenger,
        &proof,
    )
    .expect("Verifier failed");
}

#[test]
#[should_panic = "Only single-column traces are currently supported"]
fn test_prover_panics_on_wide_trace() {
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    let height = 1024;
    let width = 2; // Should panic
    let mut trace_values = vec![F::ZERO; height * width];

    for i in 0..height {
        trace_values[i * width] = F::from_usize(i);
    }
    let trace = RowMajorMatrix::new(trace_values, width);

    let code = IdentityCode { len: height };
    let hash = Keccak256Hash;
    let compress = CompressionFunctionFromHasher::new(hash);
    let serial_hasher = SerializingHasher::new(hash);
    let mmcs = MerkleTreeMmcs::<F, u8, _, _, 2, 32>::new(serial_hasher, compress, 0);
    let tensor_pcs = TensorPcs::new(code, mmcs, 40);

    let air = DummyAir;

    let mut prover_challenger = SerializingChallenger32::<F, _>::from_hasher(vec![], hash);

    let _proof = p3_multi_stark::prover::prove::<F, EF, _, _, _, _>(
        &tensor_pcs,
        &air,
        &mut prover_challenger,
        trace,
    );
}
