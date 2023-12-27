#![warn(missing_docs)]
#![allow(clippy::type_complexity)]

//! # Kalosm Language
//!
//! This crate is a collection of language processing utilities for the Kalosm project.
//!
//! There are two main traits in this crate: [`ModelExt`] for large language model (that implement [`Model`]) and [`Embedder`] for text embedding model.
//!
//! Those two traits interact with the context that this crate provides. Many different types in this crates can be converted to a [`Document`] with the [`IntoDocument`] or [`IntoDocuments`] trait:
//! - [`Page`]: Handles scraping a webpage from a request of headless browser
//! - [`SearchQuery`]: Handles searching with a search engine and scaping the result
//! - [`CrawlingCallback`]: Handles crawling a set of webpages
//! - [`FsDocument`]: Handles reading a document from the file system
//! - [`DocumentFolder`]: Handles reading an entire folder of documents from the file system
//!
//! Those documents can then be inserted into a search index:
//! - [`FuzzySearchIndex`]: A search index that performs in memory fuzzy search
//! - [`DocumentDatabase`]: A search index that performs in memory vector based search

pub mod chat;
pub mod context;
pub mod search;
pub mod task;
pub mod tool;
pub mod vector_db;

pub use kalosm_language_model;
pub use kalosm_llama;
pub use kalosm_sample;
pub use rbert;
pub use rmistral;
pub use rphi;

/// A prelude of commonly used items in kalosm-language
pub mod prelude {
    pub use crate::chat::*;
    pub use crate::context::*;
    pub use crate::task::*;
    pub use crate::tool::*;
    pub use crate::vector_db::*;
    pub use futures_util::StreamExt as _;
    pub use kalosm_language_model::*;
    pub use kalosm_llama::{Llama, LlamaBuilder, LlamaSession, LlamaSource};
    pub use kalosm_sample::*;
    pub use kalosm_streams::text_stream::*;
    pub use rbert::{Bert, BertBuilder, BertSource, BertSpace};
    pub use rmistral::{Mistral, MistralBuilder, MistralSource};
    pub use rphi::{Phi, PhiBuilder, PhiSource};
}
