//! Tensor Polynomial Commitment Scheme (PCS).
//!
//! This crate implements a polynomial commitment scheme based on linear codes
//! and multi-column commitment schemes (MMCS).

#![no_std]

#[cfg(feature = "std")]
extern crate std;

pub mod multilinear_pcs;
pub mod tensor_pcs;

pub use multilinear_pcs::*;
pub use tensor_pcs::*;
