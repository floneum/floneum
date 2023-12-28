use kalosm_language_model::{Embedder, VectorSpace};
use std::ops::Range;

use super::Chunker;
use crate::{prelude::Document, search::Chunk};

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
    pub fn chunk_str(&self, string: &str) -> Vec<Range<usize>> {
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

#[async_trait::async_trait]
impl<S: VectorSpace + Send + Sync + 'static> Chunker<S> for ChunkStrategy {
    async fn chunk<E: Embedder<S> + Send>(
        &self,
        document: &Document,
        embedder: &mut E,
    ) -> anyhow::Result<Vec<Chunk<S>>> {
        let mut chunks = Vec::new();
        let body = document.body();
        let mut documents = Vec::new();
        let chunk_ranges = self.chunk_str(body);
        for byte_range in &chunk_ranges {
            documents.push(&document.body()[byte_range.clone()]);
        }
        let embeddings = embedder.embed_batch(&documents).await?;
        for (byte_range, embedding) in chunk_ranges.into_iter().zip(embeddings) {
            chunks.push(Chunk {
                byte_range,
                embeddings: vec![embedding],
            });
        }
        Ok(chunks)
    }

    async fn chunk_batch<'a, I, E: Embedder<S> + Send>(
        &self,
        documents: I,
        embedder: &mut E,
    ) -> anyhow::Result<Vec<Vec<Chunk<S>>>>
    where
        I: IntoIterator<Item = &'a Document> + Send,
        I::IntoIter: Send,
    {
        let mut chunks = Vec::new();
        let mut chunk_strings = Vec::new();
        for document in documents {
            let body = document.body();
            let chunk = self.chunk_str(body);
            for byte_range in &chunk {
                chunk_strings.push(&body[byte_range.clone()]);
            }
            chunks.push(chunk);
        }

        let mut embeddings = embedder.embed_batch(&chunk_strings).await?;
        let mut embeddings = embeddings.drain(..);
        let mut embedded_chunks = Vec::new();

        for chunk in chunks {
            let mut document_chunks = Vec::new();
            for byte_range in chunk {
                let embedding = embeddings.next().unwrap();
                document_chunks.push(Chunk {
                    byte_range,
                    embeddings: vec![embedding],
                });
            }
            embedded_chunks.push(document_chunks);
        }

        Ok(embedded_chunks)
    }
}
