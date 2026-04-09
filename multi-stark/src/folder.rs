use p3_air::{AirBuilder, RowWindow};
use p3_field::{ExtensionField, Field};

pub struct MultilinearFolder<'a, F, EF> {
    pub main: RowWindow<'a, EF>,
    pub preprocessed: RowWindow<'a, EF>,
    pub is_first_row: EF,
    pub is_last_row: EF,
    pub is_transition: EF,
    pub alpha: EF,
    pub current_alpha: EF,
    pub accumulator: EF,
    pub _marker: core::marker::PhantomData<F>,
}

impl<'a, F, EF> AirBuilder for MultilinearFolder<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type F = F;
    type Expr = EF;
    type Var = EF;
    type MainWindow = RowWindow<'a, EF>;
    type PreprocessedWindow = RowWindow<'a, EF>;
    type PublicVar = EF; // Using EF just in case, though it can be F.

    fn main(&self) -> Self::MainWindow {
        self.main
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, _size: usize) -> Self::Expr {
        self.is_transition
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.accumulator += x.into() * self.current_alpha;
        self.current_alpha *= self.alpha;
    }
}
