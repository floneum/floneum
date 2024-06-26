#![warn(missing_docs)]
#![allow(clippy::type_complexity)]

//! # Kalosm Language
//!
//! This crate is a collection of language processing utilities for the Kalosm project.
//!
//! There are two main traits in this crate: [`prelude::ModelExt`] for large language model (that implement [`prelude::Model`]) and [`prelude::Embedder`] for text embedding model.
//!
//! Those two traits interact with the context that this crate provides. Many different types in this crates can be converted to a [`prelude::Document`] with the [`prelude::IntoDocument`] or [`prelude::IntoDocuments`] trait:
//! - [`prelude::Page`]: Handles scraping a webpage from a request of headless browser
//! - [`prelude::SearchQuery`]: Handles searching with a search engine and scaping the result
//! - [`prelude::CrawlingCallback`]: Handles crawling a set of webpages
//! - [`prelude::FsDocument`]: Handles reading a document from the file system
//! - [`prelude::DocumentFolder`]: Handles reading an entire folder of documents from the file system

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
pub use rphi;

/// A prelude of commonly used items in kalosm-language
pub mod prelude {
    pub use crate::chat::*;
    pub use crate::context::*;
    pub use crate::search::*;
    pub use crate::task::*;
    pub use crate::tool::*;
    pub use crate::vector_db::*;
    pub use futures_util::StreamExt as _;
    pub use kalosm_language_model::*;
    pub use kalosm_llama::{Llama, LlamaBuilder, LlamaSession, LlamaSource};
    pub use kalosm_sample::*;
    pub use kalosm_streams::text_stream::*;
    pub use rbert::{Bert, BertBuilder, BertSource, BertSpace};
    pub use rphi::{Phi, PhiBuilder, PhiSource};
    pub use scraper::Html;
}
