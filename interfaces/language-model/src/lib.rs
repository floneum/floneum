//! # Language Model
//!
//! This crate provides a unified interface for language models. It supports streaming text, sampling, and embedding.
//!
//! ## Usage (with the RPhi implementation crate)
//!
//! ```rust, no_run
//! use rphi::prelude::*;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut model = Phi::default();
//!     let prompt = "The capital of France is ";
//!     let mut result = model.stream_text(prompt).await.unwrap();
//!
//!     print!("{prompt}");
//!     while let Some(token) = result.next().await {
//!         print!("{token}");
//!     }
//! }
//! ```

#![warn(missing_docs)]

mod embedding;
pub use embedding::*;
mod model;
pub use model::*;
mod local;
pub use local::*;
mod remote;
pub use futures_util::StreamExt;
pub use kalosm_sample;
pub use remote::*;
mod structured;
