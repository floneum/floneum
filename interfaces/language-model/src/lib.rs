//! # Language Model
//!
//! This crate provides a unified interface for language models. It supports streaming text, sampling, and embedding.
//!
//! ## Usage (with the RPhi implementation crate)
//!
//! ```rust, no_run
//! use kalosm::language::*;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut model = Llama::phi_3().await.unwrap();
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

pub use futures_util::StreamExt;
pub use kalosm_sample;

#[cfg(feature = "remote")]
mod remote;
#[cfg(feature = "remote")]
pub use remote::*;

mod structured;
mod token_stream;
pub use token_stream::*;

mod embedding;
pub use embedding::*;
mod model;
pub use model::*;
