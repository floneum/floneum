use kalosm_language_model::{Embedder, Model, StructureParserResult, SyncModel, VectorSpace};
use kalosm_sample::{LiteralParser, ParserExt, StopOn};
use kalosm_streams::text_stream::ChannelTextStream;

use crate::{
    prelude::{Document, IndexParser, Task},
    search::Chunk,
};

use super::{ChunkStrategy, Chunker};

const TASK_DESCRIPTION: &str =
    "You generate hypothetical questions that may be answered by the given text. The questions restate any information necessary to understand the question";

const EXAMPLES: [(&str, &str); 5] = [
    ("For instance, while the chat GPT interface provides a straightforward entry point, it quickly becomes challenging to create structured workflows. Imagine wanting to search through files to find specific ones, such as all .txt files related to travel, and then upload them. With Floneum, you can achieve this seamlessly within a structured workflow, eliminating the need for manual interaction with external tools.", "What are the tradeoffs of using chat GPT?"),
    ("On the other end of the spectrum, tools like Langchain offer extensive workflow customization but come with more system requirements and potential security concerns. Langchain requires users to install tools like Python and CUDA, making it less accessible to non-developers. In addition to this, building workflows in Python code can be impractical for individuals without programming expertise. Finally, plugins in Langchain are not sandboxed, which can expose users to malware or security risks when incorporating third-party libraries.", "What are the tradeoffs of using Langchain?"),
    ("Floneum is a single executable that runs models locally, eliminating the need for complex installations. The heart of Floneum is its graph-based editor, designed to enable users without programming knowledge to build and manage their AI workflows seamlessly.", "What is Floneum?"), 
    ("Embeddings are a way to understand the meaning of text. They provide a representation of the meaning of the words used. It lets us focus on the meaning of the text instead of the specific wording of the text.", "What is an embedding?"), 
    ("While traditional databases rely on a fixed schema, NoSQL databases like MongoDB offer a flexible structure, allowing you to store and retrieve data in a more dynamic way. This flexibility is particularly beneficial for applications with evolving data requirements.", "How does MongoDB differ from traditional databases?")
];

const QUESTION_STARTERS: [&str; 9] = [
    "Who", "What", "When", "Where", "Why", "How", "Which", "Whom", "Whose",
];

const PREFIX: &str = "Questions that are answered by the previous text: ";

fn create_constraints() -> kalosm_sample::SequenceParser<
    LiteralParser<&'static str>,
    kalosm_sample::RepeatParser<
        kalosm_sample::SequenceParser<
            IndexParser<
                LiteralParser<&'static str>,
                kalosm_sample::LiteralMismatchError,
                (),
                kalosm_sample::LiteralParserOffset,
            >,
            StopOn<&'static str>,
        >,
    >,
> {
    LiteralParser::new(PREFIX).then(
        IndexParser::new(
            QUESTION_STARTERS
                .iter()
                .copied()
                .map(LiteralParser::new)
                .collect::<Vec<_>>(),
        )
        .then(StopOn::new("?").filter_characters(
            |c| matches!(c, ' ' | '?' | 'a'..='z' | 'A'..='Z' | '0'..='9' | ','),
        ))
        .repeat(1..=5),
    )
}

/// A builder for a hypothetical chunker.
pub struct HypotheticalBuilder<'a, M>
where
    M: Model,
    <M::SyncModel as SyncModel>::Session: Send,
{
    model: &'a mut M,
    task_description: Option<String>,
    examples: Option<Vec<(String, String)>>,
    chunking: Option<ChunkStrategy>,
}

impl<'a, M> HypotheticalBuilder<'a, M>
where
    M: Model,
    <M::SyncModel as SyncModel>::Session: Sync + Send,
{
    /// Set the chunking strategy.
    pub fn with_chunking(mut self, chunking: ChunkStrategy) -> Self {
        self.chunking = Some(chunking);
        self
    }

    /// Set the examples for this task. Each example should include the text and the questions that are answered by the text.
    pub fn with_examples<S: Into<String>>(
        mut self,
        examples: impl IntoIterator<Item = (S, S)>,
    ) -> HypotheticalBuilder<'a, M> {
        self.examples = Some(
            examples
                .into_iter()
                .map(|(a, b)| (a.into(), { PREFIX.to_string() + &b.into() }))
                .collect::<Vec<_>>(),
        );
        self
    }

    /// Set the task description. The task description should describe a task of generating hypothetical questions that may be answered by the given text.
    pub fn with_task_description(mut self, task_description: String) -> Self {
        self.task_description = Some(task_description);
        self
    }

    /// Build the hypothetical chunker.
    pub fn build(self) -> anyhow::Result<Hypothetical<M>> {
        let task_description = self
            .task_description
            .unwrap_or_else(|| TASK_DESCRIPTION.to_string());
        let examples = self.examples.unwrap_or_else(|| {
            EXAMPLES
                .iter()
                .map(|(a, b)| (a.to_string(), { PREFIX.to_string() + b }))
                .collect::<Vec<_>>()
        });
        let chunking = self.chunking;

        let task = Task::builder(self.model, task_description)
            .with_constraints(create_constraints())
            .with_examples(examples)
            .build();

        Ok(Hypothetical { chunking, task })
    }
}

/// Generates questions for a document.
pub struct Hypothetical<M: Model>
where
    <M::SyncModel as SyncModel>::Session: Sync + Send,
{
    chunking: Option<ChunkStrategy>,
    task:
        Task<M, StructureParserResult<ChannelTextStream<String>, ((), Vec<((usize, ()), String)>)>>,
}

impl<M: Model> Hypothetical<M>
where
    <M::SyncModel as SyncModel>::Session: Sync + Send,
{
    /// Create a new hypothetical generator.
    pub fn builder(model: &mut M) -> HypotheticalBuilder<M> {
        HypotheticalBuilder {
            model,
            task_description: None,
            examples: None,
            chunking: None,
        }
    }

    /// Generate a list of hypothetical questions about the given text.
    pub async fn generate_question(
        &self,
        text: &str,
        model: &mut M,
    ) -> anyhow::Result<Vec<String>> {
        let questions = self.task.run(text, model).await?.result().await?;
        let documents = questions
            .1
            .into_iter()
            .map(|((i, _), s)| QUESTION_STARTERS[i].to_string() + &s)
            .collect::<Vec<_>>();

        Ok(documents)
    }

    /// Turn this hypothetical generator into a chunker.
    pub fn chunker<'a>(&'a self, model: &'a mut M) -> HypotheticalChunker<'a, M> {
        HypotheticalChunker {
            hypothetical: self,
            model,
        }
    }
}

/// A hypothetical chunker.
pub struct HypotheticalChunker<'a, M: Model>
where
    <M::SyncModel as SyncModel>::Session: Sync + Send,
{
    hypothetical: &'a Hypothetical<M>,
    model: &'a mut M,
}

#[async_trait::async_trait]
impl<'a, S, M> Chunker<S> for HypotheticalChunker<'a, M>
where
    M: Model,
    <M::SyncModel as SyncModel>::Session: Sync + Send,
    S: VectorSpace + Send + Sync + 'static,
{
    async fn chunk<E: Embedder<S> + Send>(
        &mut self,
        document: &Document,
        embedder: &mut E,
    ) -> anyhow::Result<Vec<Chunk<S>>> {
        let body = document.body();

        #[allow(clippy::single_range_in_vec_init)]
        let byte_chunks = self
            .hypothetical
            .chunking
            .map(|chunking| chunking.chunk_str(body))
            .unwrap_or_else(|| vec![0..body.len()]);

        if byte_chunks.is_empty() {
            return Ok(vec![]);
        }

        let mut questions = Vec::new();
        let mut questions_count = Vec::new();
        for byte_chunk in &byte_chunks {
            let text = &body[byte_chunk.clone()];
            let mut chunk_questions = self
                .hypothetical
                .generate_question(text, self.model)
                .await?;
            questions.append(&mut chunk_questions);
            questions_count.push(chunk_questions.len());
        }
        let embeddings = embedder
            .embed_batch(&questions.iter().map(|s| s.as_str()).collect::<Vec<_>>())
            .await?;

        let mut chunks = Vec::with_capacity(embeddings.len());
        let mut questions_count = questions_count.iter();
        let mut byte_chunks = byte_chunks.into_iter();

        let mut remaining_embeddings = *questions_count.next().unwrap();
        let mut byte_chunk = byte_chunks.next().unwrap();

        for embedding in embeddings {
            while remaining_embeddings == 0 {
                if let Some(&questions_count) = questions_count.next() {
                    remaining_embeddings = questions_count;
                    byte_chunk = byte_chunks.next().unwrap();
                }
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
