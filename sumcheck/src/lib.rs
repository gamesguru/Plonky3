//! Linear-time sumcheck protocol folded over boolean hypercube.

#![no_std]

extern crate alloc;

pub mod prover;
pub mod verifier;
pub mod virtual_poly;

#[cfg(test)]
pub mod tests;
