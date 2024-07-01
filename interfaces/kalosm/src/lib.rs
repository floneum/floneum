#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub use futures_util::StreamExt as _;
pub use kalosm_streams::timed_stream::*;

#[cfg(feature = "language")]
pub mod language {
    //! Language processing utilities for the Kalosm framework.
    pub use kalosm_common::FileSource;
    pub use kalosm_language::chat::*;
    pub use kalosm_language::context::*;
    pub use kalosm_language::kalosm_language_model::{
        Embedder as _, EmbedderExt as _, Model as _, ModelExt as _, *,
    };
    pub use kalosm_language::kalosm_llama::{Llama, LlamaBuilder, LlamaSession, LlamaSource};
    pub use kalosm_language::kalosm_sample::*;
    pub use kalosm_language::prelude::Html;
    pub use kalosm_language::rbert::{Bert, BertBuilder, BertSource, BertSpace};
    pub use kalosm_language::rphi::{Phi, PhiBuilder, PhiSource};
    pub use kalosm_language::search::*;
    pub use kalosm_language::task::*;
    pub use kalosm_language::tool::*;
    pub use kalosm_language::vector_db::*;
    pub use kalosm_streams::text_stream::*;

    #[cfg(feature = "surrealdb")]
    pub use crate::surrealdb_integration::document_table::*;
}
#[cfg(feature = "sound")]
pub use kalosm_sound as audio;
#[cfg(feature = "vision")]
pub use kalosm_vision as vision;

#[cfg(feature = "language")]
mod evaluate;
#[cfg(feature = "language")]
pub use evaluate::*;

#[cfg(feature = "language")]
mod prompt_annealing;
#[cfg(feature = "language")]
pub use prompt_annealing::*;

#[cfg(feature = "surrealdb")]
mod surrealdb_integration;
#[cfg(feature = "surrealdb")]
pub use ::surrealdb;
#[cfg(feature = "surrealdb")]
pub use surrealdb_integration::*;
