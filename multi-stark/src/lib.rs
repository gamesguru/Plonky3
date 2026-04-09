//! Multilinear STARK protocol implementations orchestrating Tensor PCS and Sumcheck.

#![no_std]

extern crate alloc;

pub mod eq_poly;
pub mod folder;
pub mod proof;
pub mod prover;
pub mod verifier;
