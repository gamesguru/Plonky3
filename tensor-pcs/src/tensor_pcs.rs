extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_code::{LinearCode, SystematicCode};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use serde::{Deserialize, Serialize};

use crate::MultilinearPcs;

#[derive(Clone, Debug)]
pub struct TensorPcs<F, C, M>
where
    F: Field,
    C: LinearCode<F, RowMajorMatrix<F>>,
    M: Mmcs<F>,
{
    pub code: C,
    pub mmcs: M,
    _marker: PhantomData<F>,
}

impl<F, C, M> TensorPcs<F, C, M>
where
    F: Field,
    C: LinearCode<F, RowMajorMatrix<F>>,
    M: Mmcs<F>,
{
    pub fn new(code: C, mmcs: M) -> Self {
        Self {
            code,
            mmcs,
            _marker: PhantomData,
        }
    }
}

/// The prover data stores the original multi-linear evaluations and the MMCS prover data structure.
pub struct TensorPcsProverData<F: Field, M: Mmcs<F>> {
    pub evals: Vec<RowMajorMatrix<F>>,
    pub encoded_matrices: Vec<RowMajorMatrix<F>>,
    pub mmcs_data: M::ProverData<RowMajorMatrix<F>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TensorPcsProof<F, M>
where
    F: Field,
    M: Mmcs<F>,
{
    pub mmcs_proof: M::Proof,
    // Add additional proof details (fold vectors, etc.)
}

impl<F, C, M, Chal> MultilinearPcs<F, Chal> for TensorPcs<F, C, M>
where
    F: Field,
    C: LinearCode<F, RowMajorMatrix<F>, Out = RowMajorMatrix<F>>
        + SystematicCode<F, RowMajorMatrix<F>>,
    M: Mmcs<F>,
    Chal: ExtensionField<F>,
{
    type Commitment = M::Commitment;
    type ProverData = TensorPcsProverData<F, M>;
    type Proof = TensorPcsProof<F, M>;
    type Error = M::Error;

    fn commit(
        &self,
        evals: impl IntoIterator<Item = RowMajorMatrix<F>>,
    ) -> (Self::Commitment, Self::ProverData) {
        let evals: Vec<_> = evals.into_iter().collect();

        // 1. Encode the rows
        #[cfg(feature = "std")]
        let t0 = std::time::Instant::now();
        let mut encoded_matrices = Vec::with_capacity(evals.len());
        for e in &evals {
            let height = e.height();
            assert!(
                height.is_power_of_two(),
                "Matrix height must be a power of two"
            );
            assert_eq!(
                height,
                self.code.message_len(),
                "Matrix height must match the code's message length"
            );

            let encoded = self.code.encode_batch(e.clone());
            encoded_matrices.push(encoded);
        }
        #[cfg(feature = "std")]
        {
            let encode_dur = t0.elapsed();
        }

        // Commit to columns via MMCS
        #[cfg(feature = "std")]
        let t1 = std::time::Instant::now();
        let (commitment, mmcs_data) = self.mmcs.commit(encoded_matrices.clone());
        #[cfg(feature = "std")]
        {
            let _mmcs_dur = t1.elapsed();
            let _total = t0.elapsed();
        }

        (
            commitment,
            TensorPcsProverData {
                evals,
                encoded_matrices,
                mmcs_data,
            },
        )
    }

    fn open(
        &self,
        _prover_data: &Self::ProverData,
        _point: &[Chal],
        _challenger: &mut impl p3_challenger::FieldChallenger<F>,
    ) -> (Vec<Vec<Chal>>, Self::Proof) {
        // Implement sumcheck protocol folding over the matrix rows!
        unimplemented!("Tensor PCS opening and sumcheck logic")
    }

    fn verify(
        &self,
        _commitment: &Self::Commitment,
        _point: &[Chal],
        _values: &[Vec<Chal>],
        _proof: &Self::Proof,
        _challenger: &mut impl p3_challenger::FieldChallenger<F>,
    ) -> Result<(), Self::Error> {
        unimplemented!("Tensor PCS verification logic")
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use p3_baby_bear::BabyBear;
    use p3_code::{CodeOrFamily, IdentityCode};
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::KeccakF;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};

    use super::*;

    type F = BabyBear;
    const VECTOR_LEN: usize = p3_keccak::VECTOR_LEN;
    type MyHash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
    type MyCompress = CompressionFunctionFromHasher<MyHash, 2, 4>;
    type MyMmcs = MerkleTreeMmcs<
        [F; VECTOR_LEN],
        [u64; VECTOR_LEN],
        SerializingHasher<MyHash>,
        MyCompress,
        2,
        4,
    >;

    /// [Test] TensorPcs::commit
    /// - Construct a Keccak-based Merkle tree MMCS
    /// - Use IdentityCode as linear code
    /// - Create small multilinear eval matrix (4 row x 3 col)
    /// - Call commit & assert prover data is valid
    #[test]
    fn test_tensor_pcs_commit_identity_code() {
        let t_start = Instant::now();

        // Setup hash + compression for Merkle tree
        let hash = MyHash::new(KeccakF {});
        let compress = MyCompress::new(hash);
        let mmcs = MyMmcs::new(SerializingHasher::new(hash), compress, 0);

        // IdentityCode: message_len = codeword_len = 4 (number of rows)
        let code = IdentityCode { len: 4 };

        let pcs = TensorPcs::new(code, mmcs);

        // Create a 4x3 evaluation matrix (4 rows, 3 polynomials)
        // This represents evaluations of multilinear polys over {0,1}^2
        let values: Vec<F> = (1..=12).map(|i| F::from_u32(i)).collect();
        let evals = RowMajorMatrix::new(values.clone(), 3);

        // Commit
        let (commitment, prover_data) =
            <_ as MultilinearPcs<F, F>>::commit(&pcs, vec![evals.clone()]);

        // Verify prover data structure
        assert_eq!(prover_data.evals.len(), 1);
        assert_eq!(prover_data.evals[0].height(), 4);
        assert_eq!(prover_data.evals[0].width(), 3);

        // With IdentityCode, encoded == original
        assert_eq!(prover_data.encoded_matrices.len(), 1);
        assert_eq!(prover_data.encoded_matrices[0], evals);

        // Commitment should be non-trivial (a Merkle root hash)
        let commitment_1 = commitment;

        // Determinism: same input -> same commitment
        let (commitment_2, _) = <_ as MultilinearPcs<F, F>>::commit(&pcs, vec![evals.clone()]);
        assert_eq!(commitment_1, commitment_2);

        // Binding: different input -> different commitment
        let mut different_values = values;
        different_values[0] += F::ONE;
        let different_evals = RowMajorMatrix::new(different_values, 3);
        let (commitment_3, _) = <_ as MultilinearPcs<F, F>>::commit(&pcs, vec![different_evals]);
        assert_ne!(commitment_1, commitment_3);

        eprintln!(
            "  test_tensor_pcs_commit_identity_code took {:?}",
            t_start.elapsed()
        );
    }

    #[test]
    fn test_tensor_pcs_commit_brakedown() {
        let t_start = Instant::now();

        let hash = MyHash::new(KeccakF {});
        let compress = MyCompress::new(hash);
        let mmcs = MyMmcs::new(SerializingHasher::new(hash), compress, 0);

        let code = IdentityCode { len: 16 };
        let pcs = TensorPcs::new(code, mmcs);

        let values: Vec<F> = (0..32).map(|i| F::from_u32(i)).collect();
        let evals = RowMajorMatrix::new(values, 2); // 16 rows, 2 cols

        let (commitment, prover_data) =
            <_ as MultilinearPcs<F, F>>::commit(&pcs, vec![evals.clone()]);

        assert_eq!(prover_data.encoded_matrices[0].height(), 16);
        assert_eq!(prover_data.encoded_matrices[0].width(), 2);
        let _ = commitment;

        eprintln!(
            "  test_tensor_pcs_commit_brakedown (Identity) took {:?}",
            t_start.elapsed()
        );
    }

    /// Test committing to multiple polynomials at once
    #[test]
    fn test_tensor_pcs_commit_multiple_polys() {
        let t_start = Instant::now();

        let hash = MyHash::new(KeccakF {});
        let compress = MyCompress::new(hash);
        let mmcs = MyMmcs::new(SerializingHasher::new(hash), compress, 0);
        let code = IdentityCode { len: 8 };
        let pcs = TensorPcs::new(code, mmcs);

        // Two separate evaluation matrices, both 8 rows x 2 cols
        let vals_a: Vec<F> = (0..16).map(|i| F::from_u32(i)).collect();
        let vals_b: Vec<F> = (100..116).map(|i| F::from_u32(i)).collect();
        let mat_a = RowMajorMatrix::new(vals_a, 2);
        let mat_b = RowMajorMatrix::new(vals_b, 2);

        let (_commitment, prover_data) =
            <_ as MultilinearPcs<F, F>>::commit(&pcs, vec![mat_a, mat_b]);

        assert_eq!(prover_data.evals.len(), 2);
        assert_eq!(prover_data.encoded_matrices.len(), 2);
        assert_eq!(prover_data.evals[0].height(), 8);
        assert_eq!(prover_data.evals[1].height(), 8);

        eprintln!(
            "  test_tensor_pcs_commit_multiple_polys took {:?}",
            t_start.elapsed()
        );
    }

    #[test]
    fn test_tensor_pcs_linearity() {
        let hash = MyHash::new(KeccakF {});
        let compress = MyCompress::new(hash);
        let mmcs = MyMmcs::new(SerializingHasher::new(hash), compress, 0);

        // Use IdentityCode for a simple linearity check
        let code = IdentityCode { len: 4 };
        let pcs = TensorPcs::new(code, mmcs);

        let mat_a = RowMajorMatrix::new((1..=12).map(F::from_u32).collect(), 3);
        let mat_b = RowMajorMatrix::new((10..=21).map(F::from_u32).collect(), 3);

        // encode(A + B)
        let mut mat_sum_vals = Vec::new();
        for (a, b) in mat_a.values.iter().zip(mat_b.values.iter()) {
            mat_sum_vals.push(*a + *b);
        }
        let mat_sum = RowMajorMatrix::new(mat_sum_vals, 3);
        let encoded_sum = pcs.code.encode_batch(mat_sum);

        // encode(A) + encode(B)
        let enc_a = pcs.code.encode_batch(mat_a);
        let enc_b = pcs.code.encode_batch(mat_b);

        let mut enc_sum_expected_vals = Vec::new();
        for (a, b) in enc_a.values.iter().zip(enc_b.values.iter()) {
            enc_sum_expected_vals.push(*a + *b);
        }
        let enc_sum_expected = RowMajorMatrix::new(enc_sum_expected_vals, 3);

        assert_eq!(encoded_sum, enc_sum_expected, "Encoding should be linear");
    }
}
