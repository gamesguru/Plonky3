use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_air::Air;
use p3_air::symbolic::{AirLayout, SymbolicAirBuilder, get_max_constraint_degree_extension};
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_sumcheck::virtual_poly::VirtualPolynomial;
use p3_tensor_pcs::{StarkMultilinearPcs, TensorPcs};

use crate::eq_poly::evaluate_eq_poly_table;
use crate::folder::MultilinearFolder;

pub struct ZeroCheckVirtualPoly<'a, F, EF, A>
where
    F: Field,
    EF: ExtensionField<F>,
    A: for<'b> Air<MultilinearFolder<'b, F, EF>>,
{
    pub air: &'a A,
    pub folded_trace: Vec<Vec<EF>>,
    pub folded_shifted_trace: Vec<Vec<EF>>,
    pub eq_evals: Vec<EF>,
    pub is_first_row: Vec<EF>,
    pub is_last_row: Vec<EF>,
    pub is_transition: Vec<EF>,
    pub alpha: EF,
    pub degree: usize,
    pub _marker: PhantomData<F>,
}

impl<'a, F, EF, A> core::fmt::Debug for ZeroCheckVirtualPoly<'a, F, EF, A>
where
    F: Field,
    EF: ExtensionField<F>,
    A: for<'b> Air<MultilinearFolder<'b, F, EF>>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ZeroCheckVirtualPoly")
    }
}

impl<'a, F, EF, A> VirtualPolynomial<EF> for ZeroCheckVirtualPoly<'a, F, EF, A>
where
    F: Field,
    EF: ExtensionField<F>,
    A: for<'b> Air<MultilinearFolder<'b, F, EF>> + Sync + Send,
{
    fn num_vars(&self) -> usize {
        // Log2 of trace length
        p3_util::log2_strict_usize(self.folded_trace[0].len())
    }

    fn degree(&self) -> usize {
        self.degree
    }

    fn eval_at(&self, x: EF, index: usize) -> EF {
        let half = 1 << (self.num_vars() - 1);

        // Interpolate local & next row at x-coord
        let mut local_row = Vec::with_capacity(self.folded_trace.len());
        let mut next_row = Vec::with_capacity(self.folded_shifted_trace.len());

        for col in &self.folded_trace {
            let lo = col[index];
            let hi = col[index + half];
            local_row.push(lo + x * (hi - lo));
        }

        for col in &self.folded_shifted_trace {
            let lo = col[index];
            let hi = col[index + half];
            next_row.push(lo + x * (hi - lo));
        }

        // Interpolate eq polynomial at x-coord
        let eq_lo = self.eq_evals[index];
        let eq_hi = self.eq_evals[index + half];
        let eq_x = eq_lo + x * (eq_hi - eq_lo);

        // Interpolate other polynomials
        let is_first_row_x = self.is_first_row[index]
            + x * (self.is_first_row[index + half] - self.is_first_row[index]);
        let is_last_row_x = self.is_last_row[index]
            + x * (self.is_last_row[index + half] - self.is_last_row[index]);
        let is_transition_x = self.is_transition[index]
            + x * (self.is_transition[index + half] - self.is_transition[index]);

        // Init folder and eval AIR
        let mut folder = MultilinearFolder {
            main: p3_air::RowWindow::from_two_rows(local_row.as_slice(), next_row.as_slice()),
            preprocessed: p3_air::RowWindow::from_two_rows(&[], &[]),
            is_first_row: is_first_row_x,
            is_last_row: is_last_row_x,
            is_transition: is_transition_x,
            alpha: self.alpha,
            current_alpha: EF::ONE,
            accumulator: EF::ZERO,
            _marker: PhantomData,
        };

        self.air.eval(&mut folder);

        // Return constraints * EQ polynomial
        folder.accumulator * eq_x
    }

    fn bind(&mut self, challenge: EF) {
        let half_len = self.folded_trace[0].len() / 2;

        for col in self
            .folded_trace
            .iter_mut()
            .chain(self.folded_shifted_trace.iter_mut())
        {
            for i in 0..half_len {
                let lo = col[i];
                let hi = col[i + half_len];
                col[i] = lo + challenge * (hi - lo);
            }
            col.truncate(half_len);
        }

        for i in 0..half_len {
            let lo = self.eq_evals[i];
            let hi = self.eq_evals[i + half_len];
            self.eq_evals[i] = lo + challenge * (hi - lo);

            let lo = self.is_first_row[i];
            let hi = self.is_first_row[i + half_len];
            self.is_first_row[i] = lo + challenge * (hi - lo);

            let lo = self.is_last_row[i];
            let hi = self.is_last_row[i + half_len];
            self.is_last_row[i] = lo + challenge * (hi - lo);

            let lo = self.is_transition[i];
            let hi = self.is_transition[i + half_len];
            self.is_transition[i] = lo + challenge * (hi - lo);
        }
        self.eq_evals.truncate(half_len);
        self.is_first_row.truncate(half_len);
        self.is_last_row.truncate(half_len);
        self.is_transition.truncate(half_len);
    }
}

/// Proves execution of generic AIR constraint-matrix over Tensor PCS
pub fn prove<'a, F, EF, C, M, Challenger, A>(
    tensor_pcs: &TensorPcs<F, C, M>,
    air: &'a A,
    challenger: &mut Challenger,
    trace: p3_matrix::dense::RowMajorMatrix<F>,
) -> Result<crate::proof::MultiStarkProofForTensor<F, EF, C, M>, &'static str>
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
    // Transpose & shift trace to col vecs, O(n) cache-friendly
    let width = trace.width();
    assert_eq!(
        width, 1,
        "Only single-column traces are currently supported"
    );
    // TODO: Make `ZeroCheckVirtualPoly` accept "W" wide trace matrices by
    // packing its columns natively in `p3_air::RowWindow` slices. This allows
    // quick reuse of wide-AIR circuits (like Poseidon/Keccak)
    let height = trace.height();

    let mut folded_trace = alloc::vec![alloc::vec![EF::ZERO; height]; width];
    let mut folded_shifted_trace = alloc::vec![alloc::vec![EF::ZERO; height]; width];

    for r in 0..height {
        let next_r = (r + 1) % height;
        for c in 0..width {
            folded_trace[c][r] = EF::from(trace.values[r * width + c]);
            folded_shifted_trace[c][r] = EF::from(trace.values[next_r * width + c]);
        }
    }

    // TODO: Don't duplicate `shifted_trace` via physical memory. Save L1/L2 cache.
    // Once Plonky3's MMCS supports "strided" evals, we should replace this w/ a
    // zero-copy algebraic rotation during Sumcheck folding to halve memory need.
    let mut shifted_values = alloc::vec![F::ZERO; trace.values.len()];
    for r in 0..height {
        let next_r = (r + 1) % height;
        for c in 0..width {
            shifted_values[r * width + c] = trace.values[next_r * width + c];
        }
    }
    let shifted_trace = p3_matrix::dense::RowMajorMatrix::new(shifted_values, width);

    // Commit to BOTH matrices
    let (commit, data) = <TensorPcs<F, C, M> as StarkMultilinearPcs<F, EF>>::commit(
        tensor_pcs,
        alloc::vec![trace, shifted_trace],
    );
    p3_challenger::CanObserve::observe(challenger, commit.clone());

    // Draw random vector `eval_point`, the hypercube offset
    let num_vars = p3_util::log2_strict_usize(height);
    let mut eval_point = Vec::<EF>::with_capacity(num_vars);
    for _ in 0..num_vars {
        eval_point.push(challenger.sample_algebra_element());
    }

    // Draw random scalar & random linear combine AIR constraints
    let alpha: EF = challenger.sample_algebra_element();

    // Construct "eq polynomial eval" table
    let eq_poly_table = evaluate_eq_poly_table(&eval_point);

    // Compute AIR constraint degree dynamically.
    // We add 2: 1 for the transition/selector polynomial and 1 for the eq polynomial.
    let layout = AirLayout {
        main_width: width,
        ..Default::default()
    };
    let air_degree = get_max_constraint_degree_extension::<F, EF, _>(air, layout);
    let degree = air_degree + 2;

    // Virtual polynomial
    // TODO: `is_first_row` and `is_transition` arrays are sparse (bool vectors).
    // Instead of folding mid-loop, solve folded evals in O(1) time w/ the challenge `r`,
    // bypassing memory overhead per loop.
    let mut is_first_row = alloc::vec![EF::ZERO; height];
    is_first_row[0] = EF::ONE;
    let mut is_last_row = alloc::vec![EF::ZERO; height];
    is_last_row[height - 1] = EF::ONE;
    let mut is_transition = alloc::vec![EF::ONE; height];
    is_transition[height - 1] = EF::ZERO;

    // Construct ZeroCheck virtual polynomial: Σ_{x ∈ H} C(x) * eq(eval_point, x)
    let mut virtual_poly = ZeroCheckVirtualPoly {
        air,
        folded_trace,
        folded_shifted_trace,
        eq_evals: eq_poly_table,
        is_first_row,
        is_last_row,
        is_transition,
        alpha,
        degree,
        _marker: PhantomData,
    };

    // Run sumcheck
    let (sumcheck_proof, z) =
        p3_sumcheck::prover::SumcheckProver::prove(EF::ZERO, &mut virtual_poly, challenger);

    // Open tensor PCS at hypercube point `z`
    let (pcs_values, pcs_proof) = tensor_pcs.open(&data, &z, challenger);

    Ok(crate::proof::MultiStarkProof {
        commit,
        sumcheck_proof,
        pcs_proof,
        pcs_values,
    })
}
