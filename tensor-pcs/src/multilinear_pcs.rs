extern crate alloc;

use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// A multilinear polynomial commitment scheme.
pub trait StarkMultilinearPcs<Val: Field, Challenge: ExtensionField<Val>> {
    type Commitment: Clone + Serialize + DeserializeOwned;
    type ProverData;
    type Proof: Clone + Serialize + DeserializeOwned;
    type Error: core::fmt::Debug;

    /// Commit to batch of multilinear polynomials.
    /// `evals` is an iterator of evals of multilinear polynomials over boolean hypercube.
    /// The size of each eval must be a power of two.
    fn commit(
        &self,
        evals: impl IntoIterator<Item = p3_matrix::dense::RowMajorMatrix<Val>>,
    ) -> (Self::Commitment, Self::ProverData);

    /// Open a batch of committed polynomials at a given point, z, in the hypercube.
    fn open(
        &self,
        prover_data: &Self::ProverData,
        point: &[Challenge],
        challenger: &mut impl p3_challenger::FieldChallenger<Val>,
    ) -> (Vec<Vec<Challenge>>, Self::Proof);

    /// Verify a batch of opening proofs
    fn verify(
        &self,
        commitment: &Self::Commitment,
        point: &[Challenge],
        values: &[Vec<Challenge>],
        proof: &Self::Proof,
        challenger: &mut impl p3_challenger::FieldChallenger<Val>,
    ) -> Result<(), Self::Error>;
}
