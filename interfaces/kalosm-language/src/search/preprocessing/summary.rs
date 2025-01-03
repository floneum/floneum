use kalosm_language_model::{Embedder, Model, StructuredTextGenerationError, SyncModel};
use kalosm_sample::{LiteralParser, ParserExt};

use crate::{
    prelude::{Document, OneLine, StructuredRunner, Task},
    search::Chunk,
};

use super::{ChunkStrategy, Chunker};

const TASK_DESCRIPTION: &str = "You generate summaries of the given text.";

type Constraints = kalosm_sample::SequenceParser<LiteralParser, OneLine>;

/// Generates summaries for a document.
pub struct Summarizer {
    chunking: Option<ChunkStrategy>,
    task: Task<StructuredRunner<Constraints>>,
}

impl Summarizer {
    /// Create a new summary generator.
    pub fn new(chunking: Option<ChunkStrategy>) -> Self {
        let task = Task::builder(TASK_DESCRIPTION)
            .with_constraints(LiteralParser::new("Summary: ").then(OneLine))
            .build();
        Self { chunking, task }
    }

    /// Generate a summary for a document.
    pub async fn generate_summary<M>(
        &self,
        text: &str,
        model: &M,
    ) -> Result<Vec<String>, StructuredTextGenerationError<M::Error>>
    where
        M: Model,
        <M::SyncModel as SyncModel>::Session: Sync + Send,
        M::Error: std::fmt::Debug,
    {
        let prompt = format!("Generate a summary of the following text:\n{}", text);

        let questions = self.task.run(prompt, model).result().await?;
        let documents = vec![questions.1];

        Ok(documents)
    }

    /// Turn this summary generator into a chunker.
    pub fn summary<'a, M>(&'a self, model: &'a M) -> SummaryChunker<'a, M>
    where
        M: Model,
        <M::SyncModel as SyncModel>::Session: Sync + Send,
    {
        SummaryChunker {
            summary: self,
            model,
        }
    }
}

/// An error that can occur when chunking a document with [`SummaryChunker`].
#[derive(Debug, thiserror::Error)]
pub enum SummaryChunkerError<E1: Send + Sync + 'static, E2: Send + Sync + 'static> {
    /// An error from the text generation model.
    #[error("Text generation model error: {0}")]
    TextModelError(#[from] StructuredTextGenerationError<E1>),
    /// An error from the embedding model.
    #[error("Embedding model error: {0}")]
    EmbeddingModelError(E2),
}

/// A summary chunker.
pub struct SummaryChunker<'a, M> {
    summary: &'a Summarizer,
    model: &'a M,
}

impl<'a, M> Chunker for SummaryChunker<'a, M>
where
    M: Model,
    <M::SyncModel as SyncModel>::Session: Sync + Send,
    M::Error: std::fmt::Debug,
{
    type Error<E: Send + Sync + 'static> = SummaryChunkerError<M::Error, E>;

    async fn chunk<E: Embedder + Send>(
        &self,
        document: &Document,
        embedder: &E,
    ) -> Result<Vec<Chunk>, Self::Error<E::Error>> {
        let body = document.body();

        #[allow(clippy::single_range_in_vec_init)]
        let byte_chunks = self
            .summary
            .chunking
            .map(|chunking| chunking.chunk_str(body))
            .unwrap_or_else(|| vec![0..body.len()]);

        let mut questions = Vec::new();
        let mut questions_count = Vec::new();
        for byte_chunk in &byte_chunks {
            let text = &body[byte_chunk.clone()];
            let mut chunk_questions = self.summary.generate_summary(text, self.model).await?;
            questions.append(&mut chunk_questions);
            questions_count.push(chunk_questions.len());
        }
        let embeddings = embedder
            .embed_vec(questions)
            .await
            .map_err(SummaryChunkerError::EmbeddingModelError)?;

        let mut chunks = Vec::with_capacity(embeddings.len());
        let mut questions_count = questions_count.iter();
        let mut remaining_embeddings = *questions_count.next().unwrap();
        let mut byte_chunks = byte_chunks.into_iter();
        let mut byte_chunk = byte_chunks.next().unwrap();
        for embedding in embeddings {
            if remaining_embeddings == 0 {
                remaining_embeddings = *questions_count.next().unwrap();
                byte_chunk = byte_chunks.next().unwrap();
            }
            remaining_embeddings -= 1;
            chunks.push(Chunk {
                byte_range: byte_chunk.clone(),
                embeddings: vec![embedding],
            });
        }
        Ok(chunks)
    }
}
