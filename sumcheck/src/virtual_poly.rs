//! Virtual polynomials for sumcheck constraints.

use core::fmt::Debug;

use p3_field::Field;

/// Lazily eval'd multilinear polynomial.
pub trait VirtualPolynomial<F: Field>: Debug {
    /// Number of vars in the boolean hypercube
    fn num_vars(&self) -> usize;

    /// Algebraic degree of polynomial
    fn degree(&self) -> usize;

    /// Eval virtual polynomial at `x` for given hypercube index.
    /// Index ranges over remaining n-1 vars.
    fn eval_at(&self, x: F, index: usize) -> F;

    /// Binds largest/adjacent var to "challenge" and folds polynomial (decrease size by 1).
    fn bind(&mut self, challenge: F);
}
