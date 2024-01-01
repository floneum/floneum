use kalosm_language_model::{ChatModel, Embedder, StructureParserResult, SyncModel, VectorSpace};
use kalosm_sample::{LiteralParser, ParserExt, StopOn};
use kalosm_streams::text_stream::ChannelTextStream;

use crate::{
    prelude::{Document, Task},
    search::Chunk,
};

use super::Chunker;

const TASK_DESCRIPTION: &str =
    "You generate hypothetical questions that may be answered by the given text.";

/// Generates embeddings of questions
pub struct Hypothetical {
    task: Task<StructureParserResult<ChannelTextStream<String>, Vec<((), String)>>>,
}

impl Hypothetical {
    /// Create a new hypothetical chunker.
    pub fn new<M>(model: &mut M) -> Self
    where
        M: ChatModel,
        <M::SyncModel as SyncModel>::Session: Send,
    {
        let end_assistant_marker = model.end_assistant_marker().to_string();
        let task = Task::builder(model, TASK_DESCRIPTION)
            .with_constraints(move || {
                LiteralParser::new("Question: ")
                    .then(StopOn::new(end_assistant_marker.clone()))
                    .repeat(2..=5)
            })
            .build();
        Self { task }
    }
}

#[async_trait::async_trait]
impl<S: VectorSpace + Send + Sync + 'static> Chunker<S> for Hypothetical {
    async fn chunk<E: Embedder<S> + Send>(
        &self,
        document: &Document,
        embedder: &mut E,
    ) -> anyhow::Result<Vec<Chunk<S>>> {
        let mut chunks = Vec::new();
        let body = document.body();
        let prompt = format!(
            "Generate questions that are answered by the following text:\n{}",
            body
        );
        let questions = self.task.run(prompt).await?.result().await?;
        let documents = questions
            .iter()
            .map(|(_, text)| text.as_str())
            .collect::<Vec<_>>();
        let embeddings = embedder.embed_batch(&documents).await?;

        for embedding in embeddings {
            chunks.push(Chunk {
                byte_range: 0..body.len(),
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
        let mut texts = Vec::new();
        let mut questions = Vec::new();
        let mut document_lengths = Vec::new();
        for document in documents {
            let body = document.body();
            document_lengths.push(body.len());

            let prompt = format!(
                "Generate questions that are answered by the following text:\n{}",
                body
            );

            let question = self.task.run(prompt).await?.result().await?;
            questions.push(question.len());

            tracing::trace!("generated questions {:?}", question);

            texts.extend(question.into_iter().map(|(_, text)| text));
        }

        let mut embeddings = embedder
            .embed_batch(&texts.iter().map(|s| s.as_str()).collect::<Vec<_>>())
            .await?;
        let mut embeddings = embeddings.drain(..);
        let mut embedded_chunks = Vec::new();
        let mut document_lengths = document_lengths.iter();

        for question_count in questions {
            let mut document_chunks = Vec::new();
            let doc_len = *document_lengths.next().unwrap();
            for _ in 0..question_count {
                let embedding = embeddings.next().unwrap();
                document_chunks.push(Chunk {
                    byte_range: 0..doc_len,
                    embeddings: vec![embedding],
                });
            }
            embedded_chunks.push(document_chunks);
        }

        Ok(embedded_chunks)
    }
}
