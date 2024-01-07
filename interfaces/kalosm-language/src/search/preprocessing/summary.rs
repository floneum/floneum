use kalosm_language_model::{Embedder, Model, StructureParserResult, SyncModel, VectorSpace};
use kalosm_sample::{LiteralParser, ParserExt};
use kalosm_streams::text_stream::ChannelTextStream;

use crate::{
    prelude::{Document, Task, OneLine},
    search::Chunk,
};

use super::{ChunkStrategy, Chunker};

const TASK_DESCRIPTION: &str = "You generate summaries of the given text.";

/// Generates embeddings of questions
pub struct Summarizer {
    chunking: Option<ChunkStrategy>,
    task: Task<StructureParserResult<ChannelTextStream<String>, ((), String)>>,
}

impl Summarizer {
    /// Create a new hypothetical chunker.
    pub fn new<M>(model: &mut M, chunking: Option<ChunkStrategy>) -> Self
    where
        M: Model,
        <M::SyncModel as SyncModel>::Session: Send,
    {
        let task = Task::builder(model, TASK_DESCRIPTION)
            .with_constraints(move || {
                LiteralParser::new("Summary: ").then(OneLine)
            })
            .build();
        Self { chunking, task }
    }

    /// Generate a summary for a document.
    async fn generate_summary(&self, text: &str) -> anyhow::Result<Vec<String>> {
        let prompt = format!("Generate a summary of the following text:\n{}", text);

        let questions = self.task.run(prompt).await?.result().await?;
        let documents = vec![questions.1];

        Ok(documents)
    }
}

#[async_trait::async_trait]
impl<S: VectorSpace + Send + Sync + 'static> Chunker<S> for Summarizer {
    async fn chunk<E: Embedder<S> + Send>(
        &self,
        document: &Document,
        embedder: &mut E,
    ) -> anyhow::Result<Vec<Chunk<S>>> {
        let body = document.body();

        #[allow(clippy::single_range_in_vec_init)]
        let byte_chunks = self
            .chunking
            .map(|chunking| chunking.chunk_str(body))
            .unwrap_or_else(|| vec![0..body.len()]);

        let mut questions = Vec::new();
        let mut questions_count = Vec::new();
        for byte_chunk in &byte_chunks {
            let text = &body[byte_chunk.clone()];
            let mut chunk_questions = self.generate_summary(text).await?;
            questions.append(&mut chunk_questions);
            questions_count.push(chunk_questions.len());
        }
        let embeddings = embedder
            .embed_batch(&questions.iter().map(|s| s.as_str()).collect::<Vec<_>>())
            .await?;

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
