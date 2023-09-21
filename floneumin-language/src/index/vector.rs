use crate::{context::document::IntoDocument, index::Chunk};
use std::ops::Range;

use slab::Slab;

use crate::{
    context::document::Document,
    embedding::{Embedding, VectorSpace},
    model::Model,
    vector_db::VectorDB,
};

use super::{DocumentId, DocumentSnippet, DocumentSnippetRef, SearchIndex};

pub enum ChunkStrategy {
    Paragraph {
        paragraph_count: usize,
        overlap: usize,
    },
    Sentence {
        sentence_count: usize,
        overlap: usize,
    },
    Words {
        word_count: usize,
        overlap: usize,
    },
}

impl ChunkStrategy {
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

pub struct EmbeddedDocument<S: VectorSpace> {
    raw: Document,
    chunks: Vec<Chunk<S>>,
}

impl<S: VectorSpace> EmbeddedDocument<S> {
    pub async fn new<M: Model<S>>(raw: Document, strategy: ChunkStrategy) -> anyhow::Result<Self> {
        let ranges = strategy.chunk(raw.body());
        let mut chunks = Vec::with_capacity(ranges.len());
        for range in ranges {
            let embedding = M::embed(&raw.body()[range.clone()]).await?;
            let chunk = Chunk {
                byte_range: range,
                embedding,
            };
            chunks.push(chunk);
        }
        Ok(Self { raw, chunks })
    }
}

pub struct DocumentDatabase<S: VectorSpace, M: Model<S>> {
    documents: Slab<Document>,
    database: VectorDB<DocumentSnippet, S>,
    phantom: std::marker::PhantomData<M>,
}

#[async_trait::async_trait]
impl<M: Model<S> + Send + Sync, S: VectorSpace + Sync + Send> SearchIndex
    for DocumentDatabase<S, M>
{
    async fn add(&mut self, document: impl IntoDocument + Send + Sync) -> anyhow::Result<()> {
        let document = document.into_document().await?;
        let embedded = EmbeddedDocument::<S>::new::<M>(
            document,
            ChunkStrategy::Sentence {
                sentence_count: 5,
                overlap: 2,
            },
        )
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

    async fn search(&self, query: &str, top_n: usize) -> Vec<DocumentSnippetRef> {
        let embedding = M::embed(query).await.unwrap();
        self.search_iter(embedding, top_n).collect()
    }
}

impl<M: Model<S>, S: VectorSpace + Sync + Send> DocumentDatabase<S, M> {
    pub fn new() -> Self {
        Self {
            documents: Slab::new(),
            database: VectorDB::new(Vec::new(), Vec::new()),
            phantom: std::marker::PhantomData,
        }
    }

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
