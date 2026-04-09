use alloc::vec::Vec;

use p3_air::Air;
use p3_air::symbolic::{AirLayout, SymbolicAirBuilder, get_max_constraint_degree_extension};
use p3_field::{ExtensionField, Field};
use p3_tensor_pcs::TensorPcs;

use crate::folder::MultilinearFolder;

/// Verifies execution of a generic Air constraint matrix over Tensor-PCS.
pub fn verify<'a, F, EF, C, M, Challenger, A>(
    tensor_pcs: &TensorPcs<F, C, M>,
    air: &'a A,
    challenger: &mut Challenger,
    proof: &crate::proof::MultiStarkProofForTensor<F, EF, C, M>,
) -> Result<(), &'static str>
where
    F: Field,
    EF: ExtensionField<F> + p3_field::BasedVectorSpace<F>,
    C: p3_code::LinearCode<F, p3_matrix::dense::RowMajorMatrix<F>>
        + p3_code::SystematicCode<F, p3_matrix::dense::RowMajorMatrix<F>>,
    <C as p3_code::CodeOrFamily<F, p3_matrix::dense::RowMajorMatrix<F>>>::Out: Clone,
    M: p3_commit::Mmcs<F>,
    Challenger: p3_challenger::FieldChallenger<F>
        + p3_challenger::CanObserve<<M as p3_commit::Mmcs<F>>::Commitment>,
    A: for<'b> Air<MultilinearFolder<'b, F, EF>> + Air<SymbolicAirBuilder<F, EF>> + Sync + Send,
{
    // Recover challenge state from global commit
    p3_challenger::CanObserve::observe(challenger, proof.commit.clone());

    // Draw random vector `eval_point` which is the algebraic/combination offset of hypercube.
    let num_vars = proof.sumcheck_proof.len();
    let mut eval_point = Vec::<EF>::with_capacity(num_vars);
    for _ in 0..num_vars {
        eval_point.push(challenger.sample_algebra_element());
    }

    // Draw random scalar `alpha` to randomly linearly combine AIR constraints
    let alpha: EF = challenger.sample_algebra_element();

    // Verify sumcheck
    // Claim: zero b/c we are verifying random linear combination of trace constraints C(x) = zero
    // for all x in the hypercube.
    let width = proof.pcs_values[0].len();
    let layout = AirLayout {
        main_width: width,
        ..Default::default()
    };
    let air_degree = get_max_constraint_degree_extension::<F, EF, _>(air, layout);
    let degree = air_degree + 2;

    let (z, expected_e) = p3_sumcheck::verifier::SumcheckVerifier::verify(
        EF::ZERO,
        num_vars,
        degree,
        &proof.sumcheck_proof,
        challenger,
    )?;

    // Verify Tensor PCS opening at z
    <TensorPcs<F, C, M> as p3_tensor_pcs::MultilinearPcs<F, EF>>::verify(
        tensor_pcs,
        &proof.commit,
        &z,
        &proof.pcs_values,
        &proof.pcs_proof,
        challenger,
    )
    .map_err(|_| "Tensor PCS verification failed")?;

    // Extract evals.
    // Post-commit, vec has index 0, so 1 is next.
    let local_row = &proof.pcs_values[0];
    let next_row = &proof.pcs_values[1];

    // At z, compute MLE of is_first_row, is_last_row, is_transition
    let is_last_row = z.iter().fold(EF::ONE, |acc, x| acc * *x);
    let is_first_row = z.iter().fold(EF::ONE, |acc, x| acc * (EF::ONE - *x));
    let is_transition = EF::ONE - is_last_row;

    // Arithmetize given AIR
    let mut folder = MultilinearFolder {
        main: p3_air::RowWindow::from_two_rows(local_row.as_slice(), next_row.as_slice()),
        preprocessed: p3_air::RowWindow::from_two_rows(&[], &[]),
        is_first_row,
        is_last_row,
        is_transition,
        alpha,
        current_alpha: EF::ONE,
        accumulator: EF::ZERO,
        _marker: core::marker::PhantomData,
    };
    air.eval(&mut folder);
    // Guarantee/test of exact equivalence
    let eq_val = crate::eq_poly::eval_eq_poly(&eval_point, &z);
    let final_eval = folder.accumulator * eq_val;

    if final_eval != expected_e {
        return Err("PIOP Arithmetization error: constraints mis-match expected Sumcheck");
    }

    Ok(())
}
