use alloc::vec::Vec;

pub struct MultiStarkProof<Commit, Chal, PcsProof> {
    pub commit: Commit,
    pub sumcheck_proof: Vec<Vec<Chal>>, // Univariate rounds
    pub pcs_proof: PcsProof,            // Tensor PCS Merkle proof
    pub pcs_values: Vec<Vec<Chal>>,     // PCS opening values at `z`
}

pub type MultiStarkProofForTensor<F, EF, C, M> = MultiStarkProof<
    <M as p3_commit::Mmcs<F>>::Commitment,
    EF,
    <p3_tensor_pcs::TensorPcs<F, C, M> as p3_tensor_pcs::StarkMultilinearPcs<F, EF>>::Proof,
>;
