use crate::{IntoDocument, Document};
use floneumin_language_model::*;
use std::{
    borrow::Cow,
    fmt::Debug,
    ops::{Deref, Range},
};

mod keyword;
pub use keyword::*;
mod vector;
pub use vector::*;
mod weighted;
pub use weighted::*;

/// A document that can be added to a search index.
#[async_trait::async_trait]
pub trait IntoDocuments {
    /// Convert the document into a [`Document`]
    async fn into_documents(self) -> anyhow::Result<Vec<Document>>;
}

#[async_trait::async_trait]
impl<T: IntoDocument + Send + Sync, I> IntoDocuments for I
where
    I: IntoIterator<Item = T> + Send + Sync,
    <I as IntoIterator>::IntoIter: Send + Sync,
{
    async fn into_documents(self) -> anyhow::Result<Vec<Document>> {
        let mut documents = Vec::new();
        for document in self {
            documents.push(document.into_document().await?);
        }
        Ok(documents)
    }
}

/// A search index that can be used to search for documents.
#[async_trait::async_trait]
pub trait SearchIndex {
    /// Add a set of documents to the index
    async fn extend(&mut self, document: impl IntoDocuments + Send + Sync) -> anyhow::Result<()> {
        for document in document.into_documents().await? {
            self.add(document).await?;
        }
        Ok(())
    }
    /// Add a document to the index
    async fn add(&mut self, document: impl IntoDocument + Send + Sync) -> anyhow::Result<()>;
    /// Search the index for the given query
    async fn search(&mut self, query: &str, top_n: usize) -> Vec<DocumentSnippetRef>;
}

/// A document snippet that can be used to display a snippet of a document.
#[derive(Clone)]
pub struct Chunk<S: VectorSpace> {
    byte_range: Range<usize>,
    embedding: Embedding<S>,
}

impl<S: VectorSpace> Debug for Chunk<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Chunk")
            .field("byte_range", &self.byte_range)
            .field("embedding", &self.embedding)
            .finish()
    }
}

/// A snippet within a larger document.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentSnippet {
    document_id: DocumentId,
    byte_range: Range<usize>,
}

/// The id of a document.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentId(usize);

/// A reference to a snippet within a larger document.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentSnippetRef<'a> {
    score: f32,
    title: Cow<'a, str>,
    body: Cow<'a, str>,
    byte_range: Range<usize>,
}

impl DocumentSnippetRef<'_> {
    /// Get the title of the document
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Get the body of the document
    pub fn body(&self) -> &str {
        &self.body
    }

    /// Get the score of the document
    pub fn score(&self) -> f32 {
        self.score
    }

    /// Get the byte range this snippet covers in the original document body
    pub fn byte_range(&self) -> Range<usize> {
        self.byte_range.clone()
    }
}

impl Deref for DocumentSnippetRef<'_> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.body[self.byte_range.clone()]
    }
}
