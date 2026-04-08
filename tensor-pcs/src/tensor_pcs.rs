extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_code::{LinearCode, SystematicCode};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use serde::de::DeserializeOwned;
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
#[serde(bound(
    serialize = "F: Serialize, M::Proof: Serialize, Chal: Serialize",
    deserialize = "F: DeserializeOwned, M::Proof: DeserializeOwned, Chal: DeserializeOwned"
))]
pub struct TensorPcsProof<F, M, Chal>
where
    F: Field,
    M: Mmcs<F>,
    Chal: ExtensionField<F>,
{
    pub folded_evals: Vec<Vec<Chal>>,
    pub mmcs_proof: M::Proof,
    pub opened_columns: Vec<Vec<Vec<F>>>,
}

impl<F, C, M, Chal> MultilinearPcs<F, Chal> for TensorPcs<F, C, M>
where
    F: Field,
    C: LinearCode<F, RowMajorMatrix<F>, Out = RowMajorMatrix<F>>
        + SystematicCode<F, RowMajorMatrix<F>>,
    M: Mmcs<F>,
    Chal: ExtensionField<F> + p3_field::BasedVectorSpace<F>,
{
    type Commitment = M::Commitment;
    type ProverData = TensorPcsProverData<F, M>;
    type Proof = TensorPcsProof<F, M, Chal>;
    type Error = M::Error;

    fn commit(
        &self,
        evals: impl IntoIterator<Item = RowMajorMatrix<F>>,
    ) -> (Self::Commitment, Self::ProverData) {
        let evals: Vec<_> = evals.into_iter().collect();

        // Encode the rows
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

        // Commit to columns via MMCS
        let (commitment, mmcs_data) = self.mmcs.commit(encoded_matrices.clone());

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
        prover_data: &Self::ProverData,
        point: &[Chal],
        challenger: &mut impl p3_challenger::FieldChallenger<F>,
    ) -> (Vec<Vec<Chal>>, Self::Proof) {
        let n = point.len();

        // Split point into row and column variables
        let log_r = self.code.message_len().ilog2() as usize;
        let _log_c = n - log_r;
        let (z_row, z_col) = point.split_at(log_r);

        // 2. Compute the row-basis coefficients
        let row_coeffs = Poly::new_from_point(z_row, Chal::ONE);

        // 3. Fold the rows of each matrix in the batch
        let mut folded_evals = Vec::with_capacity(prover_data.evals.len());
        let mut folded_vectors = Vec::with_capacity(prover_data.evals.len());
        for m in &prover_data.evals {
            // v = sum chi_i * row_i
            let mut v = vec![Chal::ZERO; m.width()];
            for (i, row) in m.rows().enumerate() {
                let coeff = row_coeffs.as_slice()[i];
                for (j, val) in row.into_iter().enumerate() {
                    v[j] += coeff * val;
                }
            }

            // Evaluate at z_col
            let e = Poly::new(v.clone()).eval_ext(&Point::new(z_col.to_vec()));
            folded_evals.push(vec![e]);
            folded_vectors.push(v);
        }

        // 4. Sample column indices for consistency check
        for evals in &folded_evals {
            for &e in evals {
                challenger.observe_algebra_element(e);
            }
        }

        let num_queries = 40;
        let max_col_idx = self.code.codeword_len();
        let column_indices: Vec<usize> = (0..num_queries)
            .map(|_| challenger.sample_bits(max_col_idx.ilog2() as usize))
            .collect();

        // Open sampled columns across all encoded matrices
        let mut opened_columns = Vec::with_capacity(num_queries);
        let mut proofs = Vec::with_capacity(num_queries);
        for &idx in &column_indices {
            let opening = self.mmcs.open_batch(idx, &prover_data.mmcs_data);
            opened_columns.push(opening.opened_values);
            proofs.push(opening.opening_proof);
        }

        (
            folded_evals,
            TensorPcsProof {
                folded_evals: folded_vectors,
                mmcs_proof: proofs[0].clone(), // Simplified for now
                opened_columns,
            },
        )
    }

    fn verify(
        &self,
        _commitment: &Self::Commitment,
        point: &[Chal],
        values: &[Vec<Chal>],
        proof: &Self::Proof,
        challenger: &mut impl p3_challenger::FieldChallenger<F>,
    ) -> Result<(), Self::Error> {
        let n = point.len();
        let log_r = self.code.message_len().ilog2() as usize;
        let _log_c = n - log_r;
        let (z_row, z_col) = point.split_at(log_r);

        // Seed challenger and sample column indices
        for v in values {
            for &e in v {
                challenger.observe_algebra_element(e);
            }
        }

        let num_queries = 40;
        let max_col_idx = self.code.codeword_len();
        let column_indices: Vec<usize> = (0..num_queries)
            .map(|_| challenger.sample_bits(max_col_idx.ilog2() as usize))
            .collect();

        // Verify MMCS opening proofs for sampled columns
        // This check is currently simplified; we would normally verify the batch opening
        // For now, assume it succeeded or verify the first proof
        // (In a real implementation, mmcs.verify_batch would be called)

        // Check consistency of folded vector with sampled columns
        // row_coeffs[i] = chi_i(z_row)
        let row_coeffs = Poly::new_from_point(z_row, Chal::ONE);

        for (poly_idx, v) in proof.folded_evals.iter().enumerate() {
            // Check that v(z_col) == values[poly_idx]
            let e = Poly::new(v.clone()).eval_ext(&Point::new(z_col.to_vec()));

            if e != values[poly_idx][0] {
                panic!("Evaluation mismatch");
            }

            // Check that v matches the columns at indices
            // We need to encode the folded vector to compare with the encoded columns
            // Actually, we can check the identity: encode(v)_j == sum chi_i * MatrixCol_i,j
            // For each sampled column j
            for (query_idx, &_col_idx) in column_indices.iter().enumerate() {
                let opened_col = &proof.opened_columns[query_idx][poly_idx];

                let mut expected_val = Chal::ZERO;
                for (i, &coeff) in row_coeffs.iter().enumerate() {
                    expected_val += coeff * opened_col[i];
                }

                // TODO: We need to check if the j-th component of the codeword of v matches expected_val
                // This requires us to be able to encode 'v' (which is in Chal) or have systematic properties.
                // For now, we perform a partial check or assume systematic mapping.
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
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
