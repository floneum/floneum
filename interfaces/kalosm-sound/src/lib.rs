//! # Kalosm Sound
//!
//! This crate is a collection of audio utilities for the Kalosm project.
//!
//! There are four main parts of this crate:
//! - The [`AudioStream`] struct for streaming audio data
//! - The [`AudioBuffer`] struct for storing audio data
//! - The [`MicInput`] struct for reading audio data from a microphone
//! - The [`Whisper`] transcription model for converting audio data into text

#![warn(missing_docs)]

mod source;
pub use source::*;

pub use dasp;
pub use rodio;
pub use rwhisper::*;

mod transform;
#[allow(unused)]
pub use transform::*;
