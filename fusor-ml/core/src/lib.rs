// This crate is still heavily in development. We don't need as strict of a linting policy as the
// most of the crates in the workspace
#![allow(missing_docs)]

pub use composite::*;
pub use device::*;
pub use element_wise::CastTensor;
pub use layout::*;
pub use quantized::*;
pub use query::*;
pub use reduce::*;
pub use tensor::*;

pub(crate) use element_wise::*;
pub(crate) use matmul::*;
pub(crate) use pair_wise::*;

mod composite;
mod compute_graph;
mod device;
mod element_wise;
mod kernel;
mod layout;
mod map_layout;
mod matmul;
mod pair_wise;
mod quantized;
mod quantized_types_wgsl;
mod query;
mod reduce;
mod resize;
mod slice_assign;
mod tensor;
mod visit_tiled;
