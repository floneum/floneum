#![warn(missing_docs)]

//! # Kalosm Sound
//!
//! This crate is a collection of audio utilities for the Kalosm project.
//!
//! The central trait in this crate is the [`AsyncSource`] trait. It defines the behavior of an audio source that can be used to stream audio data. There are several implementations of this trait:
//! - Synchronous audio sources, that implement [`rodio::Source`] like files
//! - [`MicInput`], which reads audio data from a microphone
//! 
//! You can transform the audio data with:
//! - [`VoiceActivityDetectorExt::voice_activity_stream`]: Detect voice activity in the audio data
//! - [`DenoisedExt::denoise_and_detect_voice_activity`]: Denoise the audio data and detect voice activity
//! - [`AsyncSourceTranscribeExt::transcribe`]: Chunk an audio stream based on voice activity and then transcribe the chunked audio data
//! - [`VoiceActivityStreamExt::rechunk_voice_activity`]: Chunk an audio stream based on voice activity
//! - [`VoiceActivityStreamExt::filter_voice_activity`]: Filter chunks of audio data based on voice activity
//! - [`TranscribeChunkedAudioStreamExt::transcribe`]: Transcribe a chunked audio stream

mod source;
pub use source::*;

pub use dasp;
pub use rodio;
pub use rwhisper::*;

mod transform;
#[allow(unused)]
pub use transform::*;
