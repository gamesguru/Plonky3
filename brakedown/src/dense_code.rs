use alloc::vec;

use p3_code::{Code, CodeOrFamily, LinearCode, SystematicCode, SystematicCodeOrFamily};
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::stack::VerticalPair;

/// A generic systematic linear code that computes parity via a dense matrix-vector product.
///
/// This is particularly useful as the base case for Brakedown codes, where the lengths
/// are sufficiently small (e.g., $k \approx 30$) that a dense $O(k^2)$ encoding operations
/// is instantaneous and prevents restriction to 2-adic fields.
#[derive(Debug)]
pub struct DenseLinearCode<F: Field> {
    pub generator: RowMajorMatrix<F>,
    pub message_len: usize,
    pub codeword_len: usize,
}

impl<F: Field> DenseLinearCode<F> {
    pub const fn new(
        message_len: usize,
        codeword_len: usize,
        generator: RowMajorMatrix<F>,
    ) -> Self {
        Self {
            generator,
            message_len,
            codeword_len,
        }
    }
}

impl<F, In> CodeOrFamily<F, In> for DenseLinearCode<F>
where
    F: Field,
    In: Matrix<F>,
{
    type Out = VerticalPair<In, RowMajorMatrix<F>>;

    fn encode_batch(&self, messages: In) -> Self::Out {
        assert_eq!(
            messages.height(),
            self.message_len,
            "Message height mismatch"
        );

        let out_width = messages.width();
        let parity_height = self.codeword_len - self.message_len;
        let mut parity_values = vec![F::ZERO; parity_height * out_width];

        // Perform dense matrix multiplication: Parity = G * Messages
        for r in 0..parity_height {
            // Unpack row r of G
            // We use iterators if available, but RowMajorMatrix lets us just use `.values`
            // Wait, we can't assume In is Dense.
            // Let's manually perform the inner product
            for c in 0..out_width {
                let mut sum = F::ZERO;
                for i in 0..self.message_len {
                    // This could be optimized, but for base cases it's extremely small
                    let g_val = self.generator.values[r * self.message_len + i];
                    let msg_val = messages.get(i, c).unwrap_or_default();
                    sum += g_val * msg_val;
                }
                parity_values[r * out_width + c] = sum;
            }
        }

        let parity = RowMajorMatrix::new(parity_values, out_width);
        VerticalPair::new(messages, parity)
    }
}

impl<F: Field, In: Matrix<F>> Code<F, In> for DenseLinearCode<F> {
    fn message_len(&self) -> usize {
        self.message_len
    }

    fn codeword_len(&self) -> usize {
        self.codeword_len
    }
}

impl<F: Field, In: Matrix<F>> LinearCode<F, In> for DenseLinearCode<F> {}

impl<F: Field, In: Matrix<F>> SystematicCodeOrFamily<F, In> for DenseLinearCode<F> {}

impl<F: Field, In: Matrix<F>> SystematicCode<F, In> for DenseLinearCode<F> {}
