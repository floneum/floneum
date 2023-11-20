#![warn(missing_docs)]
#![allow(clippy::type_complexity)]

//! # Kalosm Learning
//!
//! This crate is a collection of teachable models for the Kalosm project.
//!
//! Supported models:
//! - [`Classifier`]

mod classifier;
pub use classifier::*;
pub use kalosm_learning_macro::*;
