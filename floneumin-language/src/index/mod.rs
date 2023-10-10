use crate::context::document::{Document, IntoDocument};
use floneumin_language_model::*;
use std::{
    borrow::Cow,
    fmt::Debug,
    ops::{Deref, Range},
};

pub mod keyword;
pub mod vector;
pub mod weighted;

#[async_trait::async_trait]
pub trait IntoDocuments {
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

#[async_trait::async_trait]
pub trait SearchIndex {
    async fn extend(&mut self, document: impl IntoDocuments + Send + Sync) -> anyhow::Result<()> {
        for document in document.into_documents().await? {
            self.add(document).await?;
        }
        Ok(())
    }
    async fn add(&mut self, document: impl IntoDocument + Send + Sync) -> anyhow::Result<()>;
    async fn search(&self, query: &str, top_n: usize) -> Vec<DocumentSnippetRef>;
}

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

#[derive(Debug, Clone, PartialEq)]
pub struct DocumentSnippet {
    document_id: DocumentId,
    byte_range: Range<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DocumentId(usize);

#[derive(Debug, Clone, PartialEq)]
pub struct DocumentSnippetRef<'a> {
    score: f32,
    title: Cow<'a, str>,
    body: Cow<'a, str>,
    byte_range: Range<usize>,
}

impl DocumentSnippetRef<'_> {
    pub fn title(&self) -> &str {
        &self.title
    }

    pub fn body(&self) -> &str {
        &self.body
    }

    pub fn score(&self) -> f32 {
        self.score
    }

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
