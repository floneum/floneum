use kalosm_language_model::{Embedder, Model, SyncModel};
use kalosm_sample::{LiteralParser, ParserExt, StopOn};

use crate::{
    prelude::{Document, IndexParser, StructuredRunner, Task},
    search::Chunk,
};

use super::{ChunkStrategy, Chunker};

const TASK_DESCRIPTION: &str =
    "You generate hypothetical questions that may be answered by the given text. The questions restate any information necessary to understand the question";

const EXAMPLES: [(&str, &str); 2] = [("A content delivery network or a CDN optimizes the distribution of web content by strategically placing servers worldwide. This reduces latency, accelerates content delivery, and enhances the overall user experience.", "What role does a content delivery network play in web performance?"), ("The Internet of Things or IoT connects everyday devices to the internet, enabling them to send and receive data. This connectivity enhances automation and allows for more efficient monitoring and control of various systems.", "What is the purpose of the Internet of Things?")];

const QUESTION_STARTERS: [&str; 9] = [
    "Who", "What", "When", "Where", "Why", "How", "Which", "Whom", "Whose",
];

const PREFIX: &str = "Questions that are answered by the previous text: ";

type Constraints = kalosm_sample::SequenceParser<
    LiteralParser,
    kalosm_sample::RepeatParser<
        kalosm_sample::SequenceParser<
            IndexParser<LiteralParser, (), kalosm_sample::LiteralParserOffset>,
            StopOn<&'static str>,
        >,
    >,
>;

fn create_constraints() -> Constraints {
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
pub struct HypotheticalBuilder {
    task_description: Option<String>,
    examples: Option<Vec<(String, String)>>,
    chunking: Option<ChunkStrategy>,
}

impl HypotheticalBuilder {
    /// Set the chunking strategy.
    pub fn with_chunking(mut self, chunking: ChunkStrategy) -> Self {
        self.chunking = Some(chunking);
        self
    }

    /// Set the examples for this task. Each example should include the text and the questions that are answered by the text.
    pub fn with_examples<S: Into<String>>(
        mut self,
        examples: impl IntoIterator<Item = (S, S)>,
    ) -> Self {
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
    pub fn build(self) -> anyhow::Result<Hypothetical> {
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

        let task = Task::builder(task_description)
            .with_constraints(create_constraints())
            .with_examples(examples)
            .build();

        Ok(Hypothetical { chunking, task })
    }
}

/// Generates questions for a document.
pub struct Hypothetical {
    chunking: Option<ChunkStrategy>,
    task: Task<StructuredRunner<Constraints>>,
}

impl Hypothetical {
    /// Create a new hypothetical generator.
    pub fn builder() -> HypotheticalBuilder {
        HypotheticalBuilder {
            task_description: None,
            examples: None,
            chunking: None,
        }
    }

    /// Generate a list of hypothetical questions about the given text.
    pub async fn generate_question<M>(&self, text: &str, model: &M) -> anyhow::Result<Vec<String>>
    where
        M: Model,
        <M::SyncModel as SyncModel>::Session: Sync + Send,
    {
        let questions = self.task.run(text, model).result().await?;
        let documents = questions
            .1
            .into_iter()
            .map(|((i, _), s)| QUESTION_STARTERS[i].to_string() + &s)
            .collect::<Vec<_>>();

        Ok(documents)
    }

    /// Turn this hypothetical generator into a chunker.
    pub fn chunker<'a, M>(&'a self, model: &'a M) -> HypotheticalChunker<'a, M>
    where
        M: Model,
        <M::SyncModel as SyncModel>::Session: Sync + Send,
    {
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
    hypothetical: &'a Hypothetical,
    model: &'a M,
}

impl<'a, M> Chunker for HypotheticalChunker<'a, M>
where
    M: Model,
    <M::SyncModel as SyncModel>::Session: Sync + Send,
{
    async fn chunk<E: Embedder + Send>(
        &self,
        document: &Document,
        embedder: &E,
    ) -> anyhow::Result<Vec<Chunk<E::VectorSpace>>> {
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
        let embeddings = embedder.embed_vec(questions).await?;

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
