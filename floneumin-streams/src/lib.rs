//! # floneumin-streams
//!
//! This crate contains utilities for handling streams of data (mainly text).

#![warn(missing_docs)]

mod sender;
pub use sender::*;
mod text_stream;
pub use text_stream::*;
mod timed_stream;
pub use timed_stream::*;
