//! # kalosm-sample
//! This is a sampling library for Kalosm.
//!
//! It handles choosing a token from a probability distribution. Samplers can be used to constrain the generation of text for example you can use a sampler to prevent the model from generating the same word twice in a row. Or you could only allow the model to generate a list of single digit numbers.

#![warn(missing_docs)]

#[doc(hidden)]
pub use anyhow;

mod structured_parser;
pub use structured_parser::*;
