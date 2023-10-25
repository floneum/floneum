//! # Floneumin Sound
//!
//! This crate is a collection of audio utilities for the Floneumin project.

#![warn(missing_docs)]

mod audio;
pub use audio::*;
mod source;
pub use rwhisper::*;
pub use source::*;
