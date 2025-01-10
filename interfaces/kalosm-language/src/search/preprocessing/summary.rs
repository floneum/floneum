use kalosm_language_model::{ChatModel, CreateChatSession, Embedder, StructuredChatModel};
use kalosm_sample::{
    CreateParserState, LiteralParser, ParseResult, ParseStatus, Parser, ParserExt,
};

use crate::{
    prelude::{Document, Task},
    search::Chunk,
};

use super::{ChunkStrategy, Chunker};

const TASK_DESCRIPTION: &str = "You generate summaries of the given text.";

type Constraints = kalosm_sample::SequenceParser<LiteralParser, OneLine>;

/// Generates summaries for a document.
pub struct Summarizer<M: CreateChatSession> {
    chunking: Option<ChunkStrategy>,
    task: Task<M>,
}

impl<M: CreateChatSession> Summarizer<M> {
    /// Create a new summary generator.
    pub fn new(chunking: Option<ChunkStrategy>, model: M) -> Self
    where
        M: ChatModel,
    {
        let task = Task::new(model, TASK_DESCRIPTION);
        Self { chunking, task }
    }

    /// Generate a summary for a document.
    pub async fn generate_summary(&self, text: &str) -> Result<Vec<String>, M::Error>
    where
        M: StructuredChatModel<Constraints> + Send + Sync + Clone + Unpin + 'static,
        M::ChatSession: Clone + Send + Sync + Unpin + 'static,
        M::Error: Send + Sync + Unpin,
    {
        let prompt = format!("Generate a summary of the following text:\n{}", text);

        let parser = LiteralParser::new("Summary: ").then(OneLine);
        let questions = self.task.run(prompt).with_constraints(parser).await?;
        let documents = vec![questions.1];

        Ok(documents)
    }
}

/// An error that can occur when chunking a document with [`SummaryChunker`].
#[derive(Debug, thiserror::Error)]
pub enum SummaryChunkerError<E1: Send + Sync + 'static, E2: Send + Sync + 'static> {
    /// An error from the text generation model.
    #[error("Text generation model error: {0}")]
    TextModelError(#[from] E1),
    /// An error from the embedding model.
    #[error("Embedding model error: {0}")]
    EmbeddingModelError(E2),
}

impl<M> Chunker for Summarizer<M>
where
    M: StructuredChatModel<Constraints> + Send + Sync + Clone + Unpin + 'static,
    M::ChatSession: Clone + Send + Sync + Unpin + 'static,
    M::Error: Send + Sync + Unpin,
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

/// One line of text with some non-whitespace characters
#[derive(Debug, Clone, Copy)]
pub(crate) struct OneLine;

/// The state of the [`OneLine`] parser
#[derive(Debug, Clone)]
pub(crate) struct OneLineState {
    all_whitespace: bool,
    bytes: Vec<u8>,
}

impl CreateParserState for OneLine {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        OneLineState {
            all_whitespace: true,
            bytes: Vec::new(),
        }
    }
}

/// An error that can occur when parsing a [`OneLine`]
#[derive(Debug, Clone)]
pub struct OneLineError;

impl std::fmt::Display for OneLineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OneLineError")
    }
}

impl std::error::Error for OneLineError {}

impl Parser for OneLine {
    type Output = String;
    type PartialState = OneLineState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> ParseResult<kalosm_sample::ParseStatus<'a, Self::PartialState, Self::Output>> {
        if input.is_empty() {
            return Ok(ParseStatus::Incomplete {
                new_state: state.clone(),
                required_next: Default::default(),
            });
        }
        let mut state = state.clone();
        let mut iter = input.iter();
        while let Some(&c) = iter.next() {
            if !c.is_ascii_alphanumeric() || matches!(c, b' ' | b'.' | b'\n') {
                kalosm_sample::bail!(OneLineError);
            }
            if state.all_whitespace {
                let c = char::from(c);
                if !c.is_whitespace() {
                    state.all_whitespace = false;
                }
            }
            if c == b'\n' {
                if state.all_whitespace {
                    kalosm_sample::bail!(OneLineError);
                } else {
                    return Ok(ParseStatus::Finished {
                        result: String::from_utf8_lossy(&state.bytes).to_string(),
                        remaining: iter.as_slice(),
                    });
                }
            }
            state.bytes.push(c);
        }
        Ok(ParseStatus::Incomplete {
            new_state: state,
            required_next: Default::default(),
        })
    }
}
