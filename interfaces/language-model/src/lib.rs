//! # Language Model
//!
//! This crate provides a unified interface for language models. It supports streaming text, sampling, and embedding.
//!
//! ## Usage (with the kalosm-llama implementation crate)
//!
//! ```rust, no_run
//! use kalosm::language::*;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut model = Llama::phi_3().await.unwrap();
//!     let prompt = "The capital of France is ";
//!     let mut result = model.complete(prompt);
//!
//!     print!("{prompt}");
//!     while let Some(token) = result.next().await {
//!         print!("{token}");
//!     }
//! }
//! ```

#![warn(missing_docs)]

use std::future::Future;

pub use futures_util::StreamExt;
pub use kalosm_sample;

#[cfg(feature = "openai")]
mod openai;
#[cfg(feature = "openai")]
pub use openai::*;
#[cfg(feature = "anthropic")]
mod claude;
#[cfg(feature = "anthropic")]
pub use claude::*;

mod embedding;
pub use embedding::*;
mod model;
pub use model::*;
mod builder;
pub use builder::*;
mod chat;
pub use chat::*;

/// A trait that is not `Send` on wasm32 targets, but is on other targets.
#[cfg(target_arch = "wasm32")]
pub trait WasmNotSend {}

/// A trait that is not `Send` on wasm32 targets, but is on other targets.
#[cfg(not(target_arch = "wasm32"))]
pub trait WasmNotSend: std::marker::Send {}

#[cfg(target_arch = "wasm32")]
impl<T> WasmNotSend for T {}

#[cfg(not(target_arch = "wasm32"))]
impl<T: std::marker::Send> WasmNotSend for T {}

/// A trait that is not `Send` or `Sync` on wasm32 targets, but is on other targets.
#[cfg(target_arch = "wasm32")]
pub trait WasmNotSendSync {}

/// A trait that is not `Send` or `Sync` on wasm32 targets, but is on other targets.
#[cfg(not(target_arch = "wasm32"))]
pub trait WasmNotSendSync: std::marker::Send + std::marker::Sync {}

#[cfg(target_arch = "wasm32")]
impl<T> WasmNotSendSync for T {}

#[cfg(not(target_arch = "wasm32"))]
impl<T: std::marker::Send + std::marker::Sync> WasmNotSendSync for T {}

/// A future that is not `Send` on wasm32 targets, but is on other targets.
pub trait FutureWasmNotSend: Future + WasmNotSend {}
impl<T: Future + WasmNotSend> FutureWasmNotSend for T {}
