use alloc::vec;

use p3_code::{Code, CodeOrFamily, LinearCode, SystematicCode, SystematicCodeOrFamily};
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::stack::VerticalPair;

/// A generic systematic linear code. Computes parity via dense matrix vector product.
///
/// Useful as the base case for Brakedown codes, where lengths are small (k ~ 30), so
/// dense $O(k^2)$ encoding operations are quick & not constrained to 2-adic fields.
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

        // Perform dense matrix multiplication: Parity = G * N_MESSAGES
        for r in 0..parity_height {
            for i in 0..self.message_len {
                let g_val = self.generator.values[r * self.message_len + i];
                // Inner loop over columns, stay locally for cache
                for c in 0..out_width {
                    let msg_val = messages.get(i, c).unwrap_or_default();
                    parity_values[r * out_width + c] += g_val * msg_val;
                }
            }
        }

        let parity = RowMajorMatrix::new(parity_values, out_width);
        VerticalPair::new(messages, parity)
    }
}

// Getters/stubs
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
