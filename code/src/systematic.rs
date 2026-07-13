use p3_field::Field;
use p3_matrix::Matrix;

use crate::{Code, CodeFamily, CodeOrFamily, LinearCode};

/// A systematic code, or a family thereof.
// TODO: Remove? Not really used.
pub trait SystematicCodeOrFamily<F: Field, In: Matrix<F>>: CodeOrFamily<F, In> {}

/// A systematic code with codeword layout `[message || parity]` (message rows first).
/// `encode_batch` must return a matrix whose first `message_len()` rows equal the input messages.
pub trait SystematicCode<F: Field, In: Matrix<F>>:
    SystematicCodeOrFamily<F, In> + Code<F, In>
{
    fn parity_len(&self) -> usize {
        self.codeword_len()
            .checked_sub(self.message_len())
            .unwrap_or_else(|| {
                panic!(
                    "SystematicCode::parity_len underflow, codeword_len = {} < message_len = {}",
                    self.codeword_len(),
                    self.message_len()
                );
            })
    }
}

pub trait SystematicLinearCode<F: Field, In: Matrix<F>>:
    SystematicCode<F, In> + LinearCode<F, In>
{
}

/// A family of systematic codes.
pub trait SystematicCodeFamily<F: Field, In: Matrix<F>>:
    SystematicCodeOrFamily<F, In> + CodeFamily<F, In>
{
}

#[cfg(test)]
mod tests {
    use p3_matrix::dense::RowMajorMatrix;
    use p3_mersenne_31::Mersenne31;

    use super::*;

    type F = Mersenne31;
    type In = RowMajorMatrix<F>;

    #[derive(Debug)]
    struct DummySystematicCode {
        message_len: usize,
        codeword_len: usize,
    }

    impl CodeOrFamily<F, In> for DummySystematicCode {
        type Out = In;
        fn encode_batch(&self, messages: In) -> Self::Out {
            messages
        }
    }

    impl Code<F, In> for DummySystematicCode {
        fn message_len(&self) -> usize {
            self.message_len
        }
        fn codeword_len(&self) -> usize {
            self.codeword_len
        }
    }

    impl SystematicCodeOrFamily<F, In> for DummySystematicCode {}
    impl SystematicCode<F, In> for DummySystematicCode {}

    #[test]
    fn test_parity_len_normal() {
        let code = DummySystematicCode {
            message_len: 4,
            codeword_len: 10,
        };
        assert_eq!(code.parity_len(), 6);
    }

    #[test]
    #[should_panic(expected = "SystematicCode::parity_len underflow")]
    fn test_parity_len_underflow() {
        let code = DummySystematicCode {
            message_len: 10,
            codeword_len: 4,
        };
        let _ = code.parity_len();
    }
}
