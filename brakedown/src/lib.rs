//! This crate contains an implementation of the Spielman-based code described in the Brakedown paper.

#![no_std]
#![allow(clippy::arithmetic_side_effects)]

extern crate alloc;

pub mod brakedown_code;
pub mod dense_code;
pub mod macros;
pub mod mul;
pub mod sparse;
pub mod standard_fast;

pub use brakedown_code::*;
pub use dense_code::*;
pub use standard_fast::*;
