#![warn(missing_docs)]

mod embedding;
pub use embedding::*;
pub use floneumin_sample;
mod model;
pub use model::*;
mod download;
mod local;
pub use futures_util::StreamExt;
