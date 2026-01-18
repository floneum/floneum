//! Composite operations that work on both CPU and GPU backends.
//!
//! These operations are built from primitive operations and work uniformly
//! across CPU and GPU tensors via the GpuOr abstraction.

mod activations;
mod construction;
mod math;
mod reductions;
