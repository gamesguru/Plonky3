extern crate alloc;

use alloc::vec;
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

use crate::StarkMultilinearPcs;

#[derive(Clone, Debug)]
pub struct TensorPcs<F, C, M>
where
    F: Field,
    C: LinearCode<F, RowMajorMatrix<F>>,
    M: Mmcs<F>,
{
    pub code: C,
    pub mmcs: M,
    pub num_queries: usize,
    _marker: PhantomData<F>,
}

impl<F, C, M> TensorPcs<F, C, M>
where
    F: Field,
    C: LinearCode<F, RowMajorMatrix<F>>,
    M: Mmcs<F>,
{
    pub fn new(code: C, mmcs: M, num_queries: usize) -> Self {
        Self {
            code,
            mmcs,
            num_queries,
            _marker: PhantomData,
        }
    }
}

/// The prover data stores the original multi-linear evaluations and the MMCS prover data structure.
pub struct TensorPcsProverData<F: Field, M: Mmcs<F>, Mat: Matrix<F>> {
    pub evals: Vec<RowMajorMatrix<F>>,
    pub encoded_matrices: Vec<Mat>,
    pub mmcs_data: M::ProverData<Mat>,
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
    pub folded_vectors: Vec<Vec<Chal>>,
    pub mmcs_proofs: Vec<M::Proof>,
    pub opened_rows: Vec<Vec<Vec<F>>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum TensorPcsError<ME> {
    MmcsError(ME),
    EvaluationMismatch,
    ConsistencyMismatch,
    InvalidProof(&'static str),
}

impl<ME: core::fmt::Display> core::fmt::Display for TensorPcsError<ME> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MmcsError(e) => write!(f, "MMCS error: {}", e),
            Self::EvaluationMismatch => write!(f, "Evaluation mismatch"),
            Self::ConsistencyMismatch => write!(f, "Consistency mismatch"),
            Self::InvalidProof(msg) => write!(f, "Invalid proof: {}", msg),
        }
    }
}

impl<F, C, M, Chal> StarkMultilinearPcs<F, Chal> for TensorPcs<F, C, M>
where
    F: Field,
    C: LinearCode<F, RowMajorMatrix<F>> + SystematicCode<F, RowMajorMatrix<F>>,
    C::Out: Clone,
    M: Mmcs<F>,
    Chal: ExtensionField<F> + p3_field::BasedVectorSpace<F>,
{
    type Commitment = M::Commitment;
    type ProverData = TensorPcsProverData<F, M, C::Out>;
    type Proof = TensorPcsProof<F, M, Chal>;
    type Error = TensorPcsError<M::Error>;

    fn commit(
        &self,
        evals: impl IntoIterator<Item = RowMajorMatrix<F>>,
    ) -> (Self::Commitment, Self::ProverData) {
        let evals: Vec<_> = evals.into_iter().collect();

        // Encode each column of each matrix
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
            // encoded is (codeword_len x width). Each column is a codeword.
            // We commit to its rows via MMCS.
            encoded_matrices.push(encoded);
        }

        // Commit via MMCS
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
        let height = self.code.message_len();
        let log_r = height.ilog2() as usize;
        let log_c = n - log_r;
        let width = 1 << log_c;

        // Variables [0..log_r] are MSBs (rows), [log_r..n] are LSBs (columns)
        let (z_row, z_col) = point.split_at(log_r);

        // Fold columns (polynomials) of each encoded matrix using z_col
        let col_coeffs_poly = Poly::new_from_point(z_col, Chal::ONE);
        let col_coeffs = col_coeffs_poly.as_slice();
        let mut folded_vectors = Vec::with_capacity(prover_data.encoded_matrices.len());
        let mut folded_evals = Vec::with_capacity(prover_data.encoded_matrices.len());

        for m in &prover_data.encoded_matrices {
            assert_eq!(m.width(), width, "Matrix width must match 2^log_c");
            // v = sum beta_j(z_col) * Col_j
            let mut v = vec![Chal::ZERO; m.height()];
            for (i, row) in m.rows().enumerate() {
                for (j, val) in row.into_iter().enumerate() {
                    v[i] += col_coeffs[j] * val;
                }
            }

            // This really shouldn't happen here
            assert!(
                v.len() >= height,
                "Encoded column shorter than message length"
            );
            // Evaluation e = v(z_row)
            // Truncate to the message part for multilinear evaluation
            let v_message = v[..height].to_vec();
            let e = Poly::new(v_message).eval_ext(&Point::new(z_row.to_vec()));
            folded_evals.push(vec![e]);
            folded_vectors.push(v);
        }

        // Observe folded vectors in Fiat-Shamir transcript
        for v in &folded_vectors {
            challenger.observe_algebra_slice(v);
        }

        // Sample row indices (indices of the codeword dimension)
        for evals in &folded_evals {
            for &e in evals {
                challenger.observe_algebra_element(e);
            }
        }

        let codeword_len = self.code.codeword_len();
        let bits = codeword_len.next_power_of_two().ilog2() as usize;
        let mut row_indices = Vec::with_capacity(self.num_queries);
        while row_indices.len() < self.num_queries {
            let idx = challenger.sample_bits(bits);
            if idx < codeword_len {
                // TODO: dedup sampled indices (!row_indices.contains(&idx)) to hit exact security targets
                row_indices.push(idx);
            }
        }

        // Open sampled rows of the encoded matrices
        let mut opened_rows = Vec::with_capacity(self.num_queries);
        let mut proofs = Vec::with_capacity(self.num_queries);
        for &idx in &row_indices {
            let opening = self.mmcs.open_batch(idx, &prover_data.mmcs_data);
            // opened_values is Vec<Vec<F>> where inner Vec is the row of width W
            opened_rows.push(opening.opened_values);
            proofs.push(opening.opening_proof);
        }

        (
            folded_evals,
            TensorPcsProof {
                folded_vectors,
                mmcs_proofs: proofs,
                opened_rows,
            },
        )
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        point: &[Chal],
        values: &[Vec<Chal>],
        proof: &Self::Proof,
        challenger: &mut impl p3_challenger::FieldChallenger<F>,
    ) -> Result<(), Self::Error> {
        let n = point.len();
        let height = self.code.message_len();
        // Validate height
        if !height.is_power_of_two() {
            return Err(TensorPcsError::InvalidProof(
                "message lengths must be powers of two",
            ));
        }
        let log_r = height.ilog2() as usize;
        if n < log_r {
            return Err(TensorPcsError::InvalidProof(
                "point length is too small for message length",
            ));
        }
        let log_c = n - log_r;
        let (z_row, z_col) = point.split_at(log_r);

        // Observe folded vectors in Fiat-Shamir transcript (must match prover)
        for v in &proof.folded_vectors {
            challenger.observe_algebra_slice(v);
        }

        for v in values {
            for &e in v {
                challenger.observe_algebra_element(e);
            }
        }

        let codeword_len = self.code.codeword_len();
        let bits = codeword_len.next_power_of_two().ilog2() as usize;
        let mut row_indices = Vec::with_capacity(self.num_queries);
        while row_indices.len() < self.num_queries {
            let idx = challenger.sample_bits(bits);
            if idx < codeword_len {
                // TODO: keep sampling logic synced with prover
                row_indices.push(idx);
            }
        }

        // Verify proof structure, avoid panics & potential malicious out-of-bounds access
        if proof.folded_vectors.len() != values.len() {
            return Err(TensorPcsError::InvalidProof(
                "folded_vectors length mismatch",
            ));
        }
        if proof.opened_rows.len() != row_indices.len() {
            return Err(TensorPcsError::InvalidProof("opened_rows length mismatch"));
        }
        if proof.mmcs_proofs.len() != row_indices.len() {
            return Err(TensorPcsError::InvalidProof("mmcs_proofs length mismatch"));
        }

        // Verify MMCS openings. Dimensions must be exact: width W = 2^log_c.
        let width = 1 << log_c;
        let dims = vec![
            p3_matrix::Dimensions {
                height: codeword_len,
                width
            };
            values.len()
        ];

        for (query_idx, &idx) in row_indices.iter().enumerate() {
            let opened_values = &proof.opened_rows[query_idx];
            let opening_proof = &proof.mmcs_proofs[query_idx];
            let opening_ref = p3_commit::BatchOpeningRef::new(opened_values, opening_proof);

            self.mmcs
                .verify_batch(commitment, &dims, idx, opening_ref)
                .map_err(TensorPcsError::MmcsError)?;
        }

        // Compute evaluation points (column coefficients for folding)
        let col_coeffs_poly = Poly::new_from_point(z_col, Chal::ONE);
        let col_coeffs = col_coeffs_poly.as_slice();

        for (poly_idx, v) in proof.folded_vectors.iter().enumerate() {
            if v.len() != codeword_len {
                return Err(TensorPcsError::InvalidProof(
                    "folded encoded column length mismatch",
                ));
            }

            // Safely return error to verifier
            if v.len() < height {
                return Err(TensorPcsError::InvalidProof(
                    "encoded column shorter than message length",
                ));
            }
            if values[poly_idx].is_empty() {
                return Err(TensorPcsError::InvalidProof("missing evaluation value"));
            }

            // Evaluation e = v(z_row)
            // Truncate to message part for multilinear evaluation
            let v_message = v[..height].to_vec();
            let e = Poly::new(v_message.clone()).eval_ext(&Point::new(z_row.to_vec()));
            if e != values[poly_idx][0] {
                return Err(TensorPcsError::EvaluationMismatch);
            }

            // Reconstruct message as RowMajorMatrix over base field
            let mut flat_coeffs = Vec::with_capacity(height * Chal::DIMENSION);
            for val in &v_message {
                // TODO: check if upstream P3 renames this to as_base_slice()
                flat_coeffs.extend_from_slice(val.as_basis_coefficients_slice());
            }
            let m = RowMajorMatrix::new(flat_coeffs, Chal::DIMENSION);
            let encoded_m = self.code.encode_batch(m);

            // Reconstruct encoded extension field elements
            let mut reencoded_v = Vec::with_capacity(codeword_len);
            for row in encoded_m.rows() {
                let row_vec: Vec<F> = row.into_iter().collect();
                let val = Chal::from_basis_coefficients_slice(&row_vec).ok_or(
                    TensorPcsError::InvalidProof("failed to reconstruct extension field element"),
                )?;
                reencoded_v.push(val);
            }

            // The folded vector from prover must be valid codeword
            if reencoded_v != *v {
                return Err(TensorPcsError::InvalidProof(
                    "folded vector is not a valid codeword",
                ));
            }

            // Linear constraint check: sum beta_j(z_col) * M_{idx, j} == v[idx]
            // where idx is sampled row of the committed matrix.
            for (query_idx, &idx) in row_indices.iter().enumerate() {
                let opened_matrix_evals = &proof.opened_rows[query_idx];
                if opened_matrix_evals.len() != values.len() {
                    return Err(TensorPcsError::InvalidProof(
                        "opened evaluations per query mismatch",
                    ));
                }

                let opened_row = &opened_matrix_evals[poly_idx];
                if opened_row.len() != width {
                    return Err(TensorPcsError::InvalidProof("opened row width mismatch"));
                }

                let mut rhs = Chal::ZERO;
                for (j, &val) in opened_row.iter().enumerate() {
                    rhs += col_coeffs[j] * val;
                }

                if rhs != v[idx] {
                    return Err(TensorPcsError::ConsistencyMismatch);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_brakedown::BrakedownCode;
    use p3_brakedown::sparse::CsrMatrix;
    use p3_challenger::SerializingChallenger32;
    use p3_code::IdentityCode;
    use p3_field::extension::BinomialExtensionField;
    use p3_keccak::Keccak256Hash;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
    use rand::SeedableRng;

    use crate::tensor_pcs::TensorPcs;
    use crate::{StarkMultilinearPcs, TensorPcsProverData};

    #[test]
    fn test_tensor_pcs_commit_identity_code() {
        let values = (1..=12).map(BabyBear::new).collect::<Vec<_>>();
        let evals = RowMajorMatrix::new(values, 3); // 4 rows, 3 columns
        let code = IdentityCode { len: 4 };

        let hash = Keccak256Hash;
        let compress = CompressionFunctionFromHasher::new(hash);
        let serial_hasher = SerializingHasher::new(hash);
        let mmcs = MerkleTreeMmcs::<BabyBear, u8, _, _, 2, 32>::new(serial_hasher, compress, 0);
        let pcs = TensorPcs::new(code, mmcs, 40);
        let (commitment, prover_data) = <TensorPcs<_, _, _> as StarkMultilinearPcs<
            BabyBear,
            BabyBear,
        >>::commit(&pcs, vec![evals.clone()]);

        let cap: p3_symmetric::MerkleCap<BabyBear, [u8; 32]> =
            p3_merkle_tree::MerkleTree::<BabyBear, u8, _, 2, 32>::cap(&prover_data.mmcs_data, 0);
        assert_eq!(commitment, cap);
    }

    #[test]
    fn test_tensor_pcs_commit_brakedown() {
        let values = (1..=32).map(BabyBear::new).collect::<Vec<_>>();
        let evals = RowMajorMatrix::new(values, 2); // 16 rows, 2 columns

        // Manual BrakedownCode initialization matching logic of brakedown! macro
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
        let a = CsrMatrix::<BabyBear>::rand_fixed_col_weight(&mut rng, 16, 16, 4);
        let b = CsrMatrix::<BabyBear>::rand_fixed_col_weight(&mut rng, 16, 16, 4);
        let inner_code = alloc::boxed::Box::new(IdentityCode { len: 16 });
        let code = BrakedownCode { a, b, inner_code };

        let hash = Keccak256Hash;
        let compress = CompressionFunctionFromHasher::new(hash);
        let serial_hasher = p3_symmetric::SerializingHasher::new(hash);
        let mmcs = MerkleTreeMmcs::<BabyBear, u8, _, _, 2, 32>::new(serial_hasher, compress, 0);

        let pcs = TensorPcs::new(code, mmcs, 40);
        let (_commitment, prover_data): (_, TensorPcsProverData<_, _, _>) =
            <TensorPcs<_, _, _> as StarkMultilinearPcs<BabyBear, BabyBear>>::commit(
                &pcs,
                vec![evals.clone()],
            );

        assert_eq!(prover_data.encoded_matrices[0].width(), 2);
    }

    #[test]
    fn test_tensor_pcs_linearity() {
        let values1 = (1..=8).map(BabyBear::new).collect::<Vec<_>>();
        let values2 = (9..=16).map(BabyBear::new).collect::<Vec<_>>();
        let matrix1 = RowMajorMatrix::new(values1, 2);
        let matrix2 = RowMajorMatrix::new(values2, 2);

        let code = IdentityCode { len: 4 };
        let hash = Keccak256Hash;
        let compress = CompressionFunctionFromHasher::new(hash);
        let serial_hasher = SerializingHasher::new(hash);
        let mmcs = MerkleTreeMmcs::<BabyBear, u8, _, _, 2, 32>::new(serial_hasher, compress, 0);
        let pcs = TensorPcs::new(code, mmcs, 40);

        let sum = matrix1
            .values
            .iter()
            .zip(&matrix2.values)
            .map(|(&a, &b)| a + b)
            .collect();
        let matrix_sum = RowMajorMatrix::new(sum, 2);

        let (_, data1) = <TensorPcs<_, _, _> as StarkMultilinearPcs<BabyBear, BabyBear>>::commit(
            &pcs,
            vec![matrix1],
        );
        let (_, data2) = <TensorPcs<_, _, _> as StarkMultilinearPcs<BabyBear, BabyBear>>::commit(
            &pcs,
            vec![matrix2],
        );
        let (_, data_sum) = <TensorPcs<_, _, _> as StarkMultilinearPcs<BabyBear, BabyBear>>::commit(
            &pcs,
            vec![matrix_sum],
        );

        for i in 0..data_sum.encoded_matrices[0].values.len() {
            assert_eq!(
                data_sum.encoded_matrices[0].values[i],
                data1.encoded_matrices[0].values[i] + data2.encoded_matrices[0].values[i]
            );
        }
    }

    #[test]
    fn test_tensor_pcs_commit_multiple_polys() {
        let v1 = (1..=4).map(BabyBear::new).collect::<Vec<_>>();
        let v2 = (5..=8).map(BabyBear::new).collect::<Vec<_>>();
        let m1 = RowMajorMatrix::new(v1, 1);
        let m2 = RowMajorMatrix::new(v2, 1);

        let code = IdentityCode { len: 4 };
        let hash = Keccak256Hash;
        let compress = CompressionFunctionFromHasher::new(hash);
        let serial_hasher = SerializingHasher::new(hash);
        let mmcs = MerkleTreeMmcs::<BabyBear, u8, _, _, 2, 32>::new(serial_hasher, compress, 0);
        let pcs = TensorPcs::new(code, mmcs, 40);

        let (_, prover_data) =
            <TensorPcs<_, _, _> as StarkMultilinearPcs<BabyBear, BabyBear>>::commit(
                &pcs,
                vec![m1, m2],
            );
        assert_eq!(prover_data.encoded_matrices.len(), 2);
    }

    #[test]
    fn test_tensor_pcs_full_protocol() {
        type F = BabyBear;
        type Chal = BinomialExtensionField<F, 4>;

        let values = (1..=16).map(F::new).collect::<Vec<_>>();
        let evals = RowMajorMatrix::new(values, 2); // 8 rows, 2 columns (2^3 x 2^1)
        let code = IdentityCode { len: 8 };

        let hash = Keccak256Hash;
        let compress = CompressionFunctionFromHasher::new(hash);
        let serial_hasher = SerializingHasher::new(hash);
        let mmcs = MerkleTreeMmcs::<F, u8, _, _, 2, 32>::new(serial_hasher, compress, 0);
        let pcs = TensorPcs::new(code, mmcs, 40);

        let (commitment, prover_data) =
            <TensorPcs<_, _, _> as StarkMultilinearPcs<F, Chal>>::commit(&pcs, vec![evals]);

        let point = vec![
            Chal::from(F::new(1)),
            Chal::from(F::new(2)),
            Chal::from(F::new(3)),
            Chal::from(F::new(4)),
        ];
        let mut challenger = SerializingChallenger32::<F, _>::from_hasher(vec![], hash);

        let (values, proof) = pcs.open(&prover_data, &point, &mut challenger);

        let mut challenger_verify = SerializingChallenger32::<F, _>::from_hasher(vec![], hash);
        let result = pcs.verify(&commitment, &point, &values, &proof, &mut challenger_verify);

        assert!(result.is_ok());
    }
}
