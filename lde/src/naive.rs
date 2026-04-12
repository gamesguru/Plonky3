use alloc::vec;

use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

use crate::UndefinedLde;

/// A naive LDE that just copies evaluations into a larger matrix.
#[derive(Clone)]
pub struct NaiveUndefinedLde;

impl<F, In> UndefinedLde<F, In> for NaiveUndefinedLde
where
    F: Field,
    In: Matrix<F>,
{
    type Out = RowMajorMatrix<F>;

    fn lde_batch(&self, polys: In, extended_height: usize) -> Self::Out {
        let original_height = polys.height();
        let original_width = polys.width();
        let mut out = RowMajorMatrix::new(
            vec![F::ZERO; original_width * extended_height],
            original_width,
        );

        for r in 0..original_height {
            for c in 0..original_width {
                out.values[r * original_width + c] = polys.get(r, c).unwrap_or_default();
            }
        }
        out
    }
}
