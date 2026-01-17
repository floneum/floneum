//! Common types for fusor tensor libraries
//!
//! This crate provides shared types used by both `fusor-core` (GPU) and `fusor-cpu` (CPU) crates.

mod layout;
mod tensor_slice;

pub use layout::*;
pub use tensor_slice::*;
