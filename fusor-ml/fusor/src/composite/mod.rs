//! Composite operations that work on both CPU and GPU backends.
//!
//! These operations are built from primitive operations and work uniformly
//! across CPU and GPU tensors via the GpuOr abstraction.

mod activations;
mod comparison;
mod construction;
mod conv;
mod index_select;
mod math;
mod normalization;
mod pool;
mod reductions;
mod rope;
mod shape;
mod where_cond;

pub use shape::{arange, arange_step, cat, stack};
