//! The index module contains different types of search indexes that can be used to search for [`crate::context::Document`]s created from [`crate::context::IntoDocument`] or [`crate::context::IntoDocuments`]

mod postprocessing;
mod preprocessing;
pub use preprocessing::*;

use kalosm_language_model::*;
use std::{fmt::Debug, ops::Range};

/// A document snippet that can be used to display a snippet of a document.
#[derive(Clone)]
pub struct Chunk {
    /// The byte range of the chunk in the original document.
    pub byte_range: Range<usize>,
    /// The embeddings of the chunk.
    pub embeddings: Vec<Embedding>,
}

impl Debug for Chunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Chunk")
            .field("byte_range", &self.byte_range)
            .field("embeddings", &self.embeddings)
            .finish()
    }
}
