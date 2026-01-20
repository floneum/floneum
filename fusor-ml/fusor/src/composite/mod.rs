//! Composite operations that work on both CPU and GPU backends.
//!
//! These operations are built from primitive operations and work uniformly
//! across CPU and GPU tensors via the GpuOr abstraction.

mod activations;
mod comparison;
mod construction;
mod conv;
mod flash_attention;
mod index_select;
mod math;
mod normalization;
mod pool;
mod reductions;
mod rope;
mod shape;
mod to_vec;
mod where_cond;

pub use shape::{arange, arange_step, cat, stack};
pub use to_vec::{ToVec1, ToVec2, ToVec3};
