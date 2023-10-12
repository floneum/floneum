use crate::index::IntoDocuments;
use crate::{context::IntoDocument, index::Chunk};
use std::ops::Range;

use floneumin_language_model::*;
use slab::Slab;

use crate::{context::Document, vector_db::VectorDB};

use super::{DocumentId, DocumentSnippet, DocumentSnippetRef, SearchIndex};

/// A strategy for chunking a document into smaller pieces.
///
/// This is used to split a document into smaller pieces to generate embeddings for each piece.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkStrategy {
    /// Split the document into paragraphs.
    Paragraph {
        /// The number of paragraphs to include in each chunk.
        paragraph_count: usize,
        /// The number of paragraphs to overlap between chunks.
        overlap: usize,
    },
    /// Split the document into sentences.
    Sentence {
        /// The number of sentences to include in each chunk.
        sentence_count: usize,
        /// The number of sentences to overlap between chunks.
        overlap: usize,
    },
    /// Split the document into words.
    Words {
        /// The number of words to include in each chunk.
        word_count: usize,
        /// The number of words to overlap between chunks.
        overlap: usize,
    },
}

impl ChunkStrategy {
    /// Chunk a string into smaller ranges.
    pub fn chunk(&self, string: &str) -> Vec<Range<usize>> {
        match self {
            Self::Paragraph {
                paragraph_count,
                overlap,
            } => {
                let mut chunks = Vec::new();
                let mut start = 0;
                let mut newline_indexes = Vec::new();
                for (i, c) in string.char_indices() {
                    if c == '\n' {
                        newline_indexes.push(i);
                        if newline_indexes.len() >= *paragraph_count {
                            if !string[start..i].trim().is_empty() {
                                chunks.push(start..i);
                            }
                            for _ in 0..*overlap {
                                start = newline_indexes.remove(0);
                            }
                        }
                    }
                }
                chunks
            }
            Self::Sentence {
                sentence_count,
                overlap,
            } => {
                let mut chunks = Vec::new();
                let mut start = 0;
                let mut sentance_start_indexes = Vec::new();
                for (i, c) in string.char_indices() {
                    if c == '.' {
                        sentance_start_indexes.push(i);
                        if sentance_start_indexes.len() >= *sentence_count {
                            if !string[start..i].trim().is_empty() {
                                chunks.push(start..i);
                            }
                            for _ in 0..*overlap {
                                start = sentance_start_indexes.remove(0);
                            }
                        }
                    }
                }
                chunks
            }
            Self::Words {
                word_count,
                overlap,
            } => {
                let mut chunks = Vec::new();
                let mut start = 0;
                let mut word_start_indexes = Vec::new();
                for (i, c) in string.char_indices() {
                    if c == ' ' {
                        word_start_indexes.push(i);
                        if word_start_indexes.len() >= *word_count {
                            if !string[start..i].trim().is_empty() {
                                chunks.push(start..i);
                            }
                            for _ in 0..*overlap {
                                start = word_start_indexes.remove(0);
                            }
                        }
                    }
                }
                chunks
            }
        }
    }
}

impl Default for ChunkStrategy {
    fn default() -> Self {
        Self::Paragraph {
            paragraph_count: 3,
            overlap: 1,
        }
    }
}

/// A document that has been split into smaller chunks and embedded.
pub struct EmbeddedDocument<S: VectorSpace> {
    raw: Document,
    chunks: Vec<Chunk<S>>,
}

impl<S: VectorSpace> std::fmt::Debug for EmbeddedDocument<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddedDocument")
            .field("raw", &self.raw)
            .field("chunks", &self.chunks)
            .finish()
    }
}

impl<S: VectorSpace + Send + Sync + 'static> EmbeddedDocument<S> {
    /// Create a new embedded document.
    pub async fn new<M: Embedder<S>>(
        embedder: &mut M,
        raw: Document,
        strategy: ChunkStrategy,
    ) -> anyhow::Result<Self> {
        let mut chunks = Vec::new();
        let body = raw.body();
        let mut documents = Vec::new();
        let chunk_ranges = strategy.chunk(body);
        for byte_range in &chunk_ranges {
            documents.push(&raw.body()[byte_range.clone()]);
        }
        let embeddings = embedder.embed_batch(&documents).await?;
        for (byte_range, embedding) in chunk_ranges.into_iter().zip(embeddings) {
            chunks.push(Chunk {
                byte_range,
                embedding,
            });
        }
        Ok(Self { raw, chunks })
    }

    /// Create a batch of embedded documents.
    pub async fn batch_new<M: Embedder<S>>(
        embedder: &mut M,
        raw: Vec<Document>,
        strategy: ChunkStrategy,
    ) -> anyhow::Result<Vec<Self>> {
        let mut chunks = Vec::new();
        let mut documents = Vec::new();
        for document in &raw {
            let body = document.body();
            let chunk = strategy.chunk(body);
            for byte_range in &chunk {
                documents.push(&body[byte_range.clone()]);
            }
            chunks.push(chunk);
        }

        let mut embeddings = embedder.embed_batch(&documents).await?;
        let mut embeddings = embeddings.drain(..).rev();
        let mut documents = Vec::new();

        for (raw, chunk) in raw.into_iter().zip(chunks) {
            let mut document_chunks = Vec::new();
            for byte_range in chunk {
                let embedding = embeddings.next().unwrap();
                document_chunks.push(Chunk {
                    byte_range,
                    embedding,
                });
            }
            documents.push(Self {
                raw,
                chunks: document_chunks,
            });
        }

        Ok(documents)
    }
}

/// A document database that stores documents in a [`VectorDB`] in [`Chunk`]s.
///
/// The documents can be searched using the [`SearchIndex`] trait. This database will search based on each chunks embedding to find documents with a similar meaning.
pub struct DocumentDatabase<S: VectorSpace + Send + Sync + 'static, M: Embedder<S>> {
    embedder: M,
    documents: Slab<Document>,
    database: VectorDB<DocumentSnippet, S>,
    strategy: ChunkStrategy,
}

#[async_trait::async_trait]
impl<M: Embedder<S> + Send + Sync + 'static, S: VectorSpace + Sync + Send + 'static> SearchIndex
    for DocumentDatabase<S, M>
{
    async fn add(&mut self, document: impl IntoDocument + Send + Sync) -> anyhow::Result<()> {
        let document = document.into_document().await?;
        let embedded = EmbeddedDocument::<S>::new(&mut self.embedder, document, self.strategy)
            .await
            .unwrap();
        let id = self.documents.insert(embedded.raw);
        for chunk in embedded.chunks {
            let snippet = DocumentSnippet {
                document_id: DocumentId(id),
                byte_range: chunk.byte_range,
            };
            self.database.add_embedding(chunk.embedding, snippet);
        }

        Ok(())
    }

    async fn extend(&mut self, documents: impl IntoDocuments + Send + Sync) -> anyhow::Result<()> {
        let documents = documents.into_documents().await?;
        let embedded =
            EmbeddedDocument::<S>::batch_new(&mut self.embedder, documents, self.strategy)
                .await
                .unwrap();
        let mut embeddings = Vec::new();
        let mut values = Vec::new();
        for embedded in embedded {
            let id = self.documents.insert(embedded.raw);
            for chunk in embedded.chunks {
                let snippet = DocumentSnippet {
                    document_id: DocumentId(id),
                    byte_range: chunk.byte_range,
                };
                embeddings.push(chunk.embedding);
                values.push(snippet);
            }
        }
        self.database.add_embeddings(embeddings, values);

        Ok(())
    }

    async fn search(&mut self, query: &str, top_n: usize) -> Vec<DocumentSnippetRef> {
        let embedding = self.embedder.embed(query).await.unwrap();
        self.search_iter(embedding, top_n).collect()
    }
}

impl<M: Embedder<S>, S: VectorSpace + Sync + Send + 'static> DocumentDatabase<S, M> {
    /// Create a new document database.
    pub fn new(embedder: M, chunk_strategy: ChunkStrategy) -> Self {
        Self {
            documents: Slab::new(),
            database: VectorDB::new(Vec::new(), Vec::new()),
            strategy: chunk_strategy,
            embedder,
        }
    }

    /// Find the closest documents to a given embedding.
    pub fn search_iter(
        &self,
        embedding: Embedding<S>,
        n: usize,
    ) -> impl Iterator<Item = DocumentSnippetRef<'_>> {
        self.database
            .get_closest(embedding, n)
            .into_iter()
            .map(|(score, snippet)| {
                let document = &self.documents[snippet.document_id.0];
                DocumentSnippetRef {
                    score,
                    title: document.title().into(),
                    body: document.body().into(),
                    byte_range: snippet.byte_range.clone(),
                }
            })
    }

    /// Find the documents within a given distance of an embedding.
    pub fn get_within_iter(
        &self,
        embedding: Embedding<S>,
        distance: f32,
    ) -> impl Iterator<Item = DocumentSnippetRef<'_>> {
        self.database
            .get_within(embedding, distance)
            .into_iter()
            .map(|(score, snippet)| {
                let document = &self.documents[snippet.document_id.0];
                DocumentSnippetRef {
                    score,
                    title: document.title().into(),
                    body: document.body().into(),
                    byte_range: snippet.byte_range.clone(),
                }
            })
    }
}
