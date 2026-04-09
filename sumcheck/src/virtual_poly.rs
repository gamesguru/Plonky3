//! Virtual polynomials for lazy constraint eval during sumcheck.

use core::fmt::Debug;

use p3_field::Field;

/// A trait: lazily eval'd multilinear polynomial, "ZeroCheck"
pub trait VirtualPolynomial<F: Field>: Debug + Sync + Send {
    /// Number of vars in boolean hypercube
    fn num_vars(&self) -> usize;

    /// Algebraic degree of polynomial, which determines univarate degree sent each sumcheck round
    fn degree(&self) -> usize;

    /// Evaluates virtual polynomial lazily for a specific `x` coord of current round variable and a
    /// specific bit-index across the remaining hypercube.
    /// We calculate this natively on-the-fly, construct the initial univariate polynomial evals,
    /// and NEVER allocate the huge `[F; 2^n]` constraints array.
    fn eval_at(&self, x: F, index: usize) -> F;

    /// Folds underlying multilinear polynomials, decreasing the number of vars by 1.
    fn bind(&mut self, challenge: F);
}
