//! # Floneumin Language
//!
//! This crate is a collection of language processing utilities for the Floneumin project.

#![warn(missing_docs)]

mod context;
pub use context::*;
mod index;
pub use index::*;
mod tool;
pub use tool::*;
mod vector_db;
pub use floneumin_sample;
pub use vector_db::*;
