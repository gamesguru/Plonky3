#![allow(clippy::arithmetic_side_effects)]

use alloc::vec;

use p3_field::{Field, add_scaled_slice_in_place};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::sparse::CsrMatrix;

/// Compute `C = A * B`, where `A` in a CSR matrix and `B` is a dense matrix.
///
/// # Panics
/// Panics if dimensions of input matrices don't match.
pub fn mul_csr_dense<F, B>(a: &CsrMatrix<F>, b: &B) -> RowMajorMatrix<F>
where
    F: Field,
    B: Matrix<F> + Sync,
{
    assert_eq!(a.width(), b.height(), "A, B dimensions don't match");
    let c_width = b.width();

    let mut c_values = vec![F::ZERO; a.height() * c_width];
    c_values
        .par_chunks_mut(c_width)
        .enumerate()
        .for_each(|(a_row_idx, c_row)| {
            for &(a_col_idx, a_val) in a.sparse_row(a_row_idx) {
                let b_row = b
                    .row_slice(a_col_idx)
                    .expect("index within validated bounds");
                add_scaled_slice_in_place(c_row, &b_row, a_val);
            }
        });
    RowMajorMatrix::new(c_values, c_width)
}
