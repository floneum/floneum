#![warn(missing_docs)]

//! # Floneumin Language
//!
//! This crate is a collection of language processing utilities for the Floneumin project.
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

mod context;
pub use context::*;
mod index;
pub use index::*;
mod tool;
pub use tool::*;
mod vector_db;
pub use kalosm_language_model::*;
pub use rbert::{Bert, BertBuilder, BertSource};
pub use rmistral::{Mistral, MistralBuilder, MistralSource};
pub use rphi::{Phi, PhiBuilder, PhiSource};
pub use vector_db::*;
