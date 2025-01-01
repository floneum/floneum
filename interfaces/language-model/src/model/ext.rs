use crate::structured::generate_structured;
use crate::{TokenOutputStream, TokenOutputStreamError};
use futures_util::{Future, FutureExt};
use futures_util::{Stream, StreamExt};
use kalosm_sample::{Schema, StopOn};
use kalosm_sample::{CreateParserState, Parse};
use kalosm_sample::{LiteralParser, Parser};
use kalosm_streams::text_stream::ChannelTextStream;
use llm_samplers::configure::SamplerChainBuilder;
use llm_samplers::prelude::*;
use std::any::Any;
use std::convert::Infallible;
use std::fmt::Display;
use std::future::IntoFuture;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::task::Poll;
use thiserror::Error;
use tokenizers::tokenizer::Tokenizer;

use super::generation_parameters::GenerationParameters;

/// A builder for the [`ModelExt::stream_text`] method.
pub struct StreamTextBuilder<'a, M: Model> {
    self_: &'a M,
    prompt: &'a str,
    parameters: GenerationParameters,
    #[allow(clippy::type_complexity)]
    future: fn(
        &'a M,
        &'a str,
        GenerationParameters,
    ) -> Pin<
        Box<dyn std::future::Future<Output = Result<M::TextStream, M::Error>> + Send + 'a>,
    >,
}

impl<'a, M: Model> StreamTextBuilder<'a, M> {
    /// Create a new builder to return from the [`ModelExt::stream_text`] method.
    #[allow(clippy::type_complexity)]
    pub fn new(
        prompt: &'a str,
        self_: &'a M,
        future: fn(
            &'a M,
            &'a str,
            GenerationParameters,
        ) -> Pin<
            Box<dyn std::future::Future<Output = Result<M::TextStream, M::Error>> + Send + 'a>,
        >,
    ) -> Self {
        Self {
            self_,
            prompt,
            parameters: GenerationParameters::default(),
            future,
        }
    }

    /// Set the generation parameters to use when generating text. This will override any parameters set by other methods.
    pub fn with_generation_parameters(mut self, parameters: GenerationParameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Set the temperature to use when generating text.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.parameters.temperature = temperature;
        self
    }

    /// Set the mu to use when generating text.
    pub fn with_mu(mut self, mu: f32) -> Self {
        self.parameters.mu = mu;
        self
    }

    /// Set the tau to use when generating text.
    pub fn with_tau(mut self, tau: f32) -> Self {
        self.parameters.tau = tau;
        self
    }

    /// Set the eta to use when generating text.
    pub fn with_eta(mut self, eta: f32) -> Self {
        self.parameters.eta = eta;
        self
    }

    /// Set the repetition penalty to use when generating text.
    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.parameters.repetition_penalty = repetition_penalty;
        self
    }

    /// Set the repetition penalty range to use when generating text.
    pub fn with_repetition_penalty_range(mut self, repetition_penalty_range: u32) -> Self {
        self.parameters.repetition_penalty_range = repetition_penalty_range;
        self
    }

    /// Set the maximum length to use when generating text.
    pub fn with_max_length(mut self, max_length: u32) -> Self {
        self.parameters.max_length = max_length;
        self
    }

    /// Set the string to stop on when generating text.
    pub fn with_stop_on(mut self, stop_on: impl Into<Option<String>>) -> Self {
        self.parameters.stop_on = stop_on.into();
        self
    }
}

impl<'a, M: Model> IntoFuture for StreamTextBuilder<'a, M> {
    type Output = Result<M::TextStream, M::Error>;
    type IntoFuture = Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        let Self {
            self_,
            prompt,
            parameters,
            future,
        } = self;
        future(self_, prompt, parameters)
    }
}

/// A builder for the [`ModelExt::generate_text`] method.
#[allow(clippy::type_complexity)]
pub struct GenerateTextBuilder<'a, M: Model> {
    self_: &'a M,
    prompt: &'a str,
    parameters: GenerationParameters,
    future: fn(
        &'a M,
        &'a str,
        GenerationParameters,
    )
        -> Pin<Box<dyn std::future::Future<Output = Result<String, M::Error>> + Send + 'a>>,
}

impl<'a, M: Model> GenerateTextBuilder<'a, M> {
    /// Create a new builder to return from the [`ModelExt::generate_text`] method.
    #[allow(clippy::type_complexity)]
    pub fn new(
        prompt: &'a str,
        self_: &'a M,
        future: fn(
            &'a M,
            &'a str,
            GenerationParameters,
        ) -> Pin<
            Box<dyn std::future::Future<Output = Result<String, M::Error>> + Send + 'a>,
        >,
    ) -> Self {
        Self {
            self_,
            prompt,
            parameters: GenerationParameters::default(),
            future,
        }
    }

    /// Set the generation parameters to use when generating text. This will override any parameters set by other methods.
    pub fn with_generation_parameters(mut self, parameters: GenerationParameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Set the temperature to use when generating text.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.parameters.temperature = temperature;
        self
    }

    /// Set the mu to use when generating text.
    pub fn with_mu(mut self, mu: f32) -> Self {
        self.parameters.mu = mu;
        self
    }

    /// Set the tau to use when generating text.
    pub fn with_tau(mut self, tau: f32) -> Self {
        self.parameters.tau = tau;
        self
    }

    /// Set the eta to use when generating text.
    pub fn with_eta(mut self, eta: f32) -> Self {
        self.parameters.eta = eta;
        self
    }

    /// Set the repetition penalty to use when generating text.
    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.parameters.repetition_penalty = repetition_penalty;
        self
    }

    /// Set the repetition penalty range to use when generating text.
    pub fn with_repetition_penalty_range(mut self, repetition_penalty_range: u32) -> Self {
        self.parameters.repetition_penalty_range = repetition_penalty_range;
        self
    }

    /// Set the maximum length to use when generating text.
    pub fn with_max_length(mut self, max_length: u32) -> Self {
        self.parameters.max_length = max_length;
        self
    }

    /// Set the string to stop on when generating text.
    pub fn with_stop_on(mut self, stop_on: impl Into<Option<String>>) -> Self {
        self.parameters.stop_on = stop_on.into();
        self
    }
}

impl<'a, M: Model> IntoFuture for GenerateTextBuilder<'a, M> {
    type Output = Result<String, M::Error>;
    type IntoFuture = Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        let Self {
            self_,
            prompt,
            parameters,
            future,
        } = self;
        future(self_, prompt, parameters)
    }
}

#[doc = include_str!("../docs/model.md")]
pub trait ModelExt: Model + Send + Sync + 'static {
    /// Generate text with the given prompt. This function generates a builder with extra parameters that can be set. To execute the builder, just call `await` on it.
    ///
    /// ```rust, no_run
    /// use kalosm::language::*;
    /// use std::io::Write;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let mut model = Llama::new().await.unwrap();
    ///     let prompt = "The capital of France is";
    ///     let mut result = model.generate_text(prompt).with_max_length(300).await.unwrap();
    ///
    ///     println!("{prompt}{result}");
    /// }
    /// ```
    fn generate_text<'a>(&'a self, prompt: &'a str) -> GenerateTextBuilder<'a, Self>
    where
        Self: Sized,
    {
        GenerateTextBuilder::new(prompt, self, |self_, prompt, generation_parameters| {
            Box::pin(async move {
                self_
                    .generate_text_inner(prompt, generation_parameters)
                    .await
            })
        })
    }

    /// Generate text with the given prompt. This function generates a builder with extra parameters that can be set. To execute the builder, just call `await` on it.
    ///
    /// ```rust, no_run
    /// use kalosm::language::*;
    /// use std::io::Write;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let mut model = Llama::new().await.unwrap();
    ///     let prompt = "The capital of France is";
    ///     let mut result = model.stream_text(prompt).with_max_length(300).await.unwrap();
    ///
    ///     print!("{prompt}");
    ///     while let Some(token) = result.next().await {
    ///         print!("{token}");
    ///         std::io::stdout().flush().unwrap();
    ///     }
    /// }
    /// ```
    fn stream_text<'a>(&'a self, prompt: &'a str) -> StreamTextBuilder<'a, Self>
    where
        Self: Sized,
    {
        StreamTextBuilder::new(prompt, self, |self_, prompt, generation_parameters| {
            Box::pin(async move { self_.stream_text_inner(prompt, generation_parameters).await })
        })
    }

    /// Run some code synchronously with the model.
    ///
    /// # Example
    /// ```rust, no_run
    /// use kalosm::language::*;
    /// use kalosm_language_model::Model;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let mut llm = Llama::new().await.unwrap();
    ///
    ///     let tokenizer = llm.tokenizer();
    ///     // Start a sync task on the model
    ///     llm.run_sync(move |llm: &mut <Llama as Model>::SyncModel| {
    ///         Box::pin(async move {
    ///             let question = "What is 10 + 10?";
    ///
    ///             // Create a new session of the model
    ///             let mut session = llm.new_session().unwrap();
    ///
    ///             // Feed the question into the model
    ///             let mut logits = Vec::new();
    ///             llm.feed_text(&mut session, question, &mut logits).unwrap();
    ///
    ///             println!("logits: {:?}", logits);
    ///         })
    ///     })
    ///     .unwrap();
    /// }
    /// ```
    fn run_sync(
        &self,
        f: impl for<'a> FnOnce(
                &'a mut Self::SyncModel,
            ) -> Pin<Box<dyn std::future::Future<Output = ()> + 'a>>
            + Send
            + 'static,
    ) -> Result<(), Self::Error> {
        self.run_sync_raw(Box::new(f))
    }

    /// Generate a type that implements [`Parse`] with the given prompt.
    ///
    /// # Example
    /// ```rust, no_run
    /// use kalosm::language::*;
    ///
    /// // You can derive an efficient parser for your struct with the `Parse` trait
    /// #[derive(Parse, Clone, Debug)]
    /// struct Account {
    ///     username: String,
    ///     age: u8,
    /// }
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<()> {
    /// let llm = Llama::new().await?;
    /// let prompt = "A list of accounts with random realistic usernames and ages in JSON format: ";
    ///
    /// let accounts: [Account; 10] = llm.generate_parsed(prompt).await?;
    /// println!("{:#?}", accounts);
    /// # Ok(())
    /// # }
    /// ```
    fn generate_parsed<P: Parse + 'static>(
        &self,
        prompt: &str,
    ) -> StructureParserResult<Self::TextStream, P, StructuredTextGenerationError<Self::Error>>
    where
        Self::TextStream: From<tokio::sync::mpsc::UnboundedReceiver<String>>,
    {
        self.stream_structured_text(prompt, P::new_parser())
    }

    /// Generate structured text with the given prompt and constraints.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let llm = Llama::new().await.unwrap();
    ///
    /// #[derive(Debug, Clone, Parse)]
    /// enum Size {
    ///     Small,
    ///     Medium,
    ///     Large,
    /// }
    ///
    /// let size = llm.stream_structured_text("A elephant is ", Size::new_parser()).await.unwrap();
    /// println!("{size:?}");
    /// # }
    /// ```
    fn stream_structured_text<P>(
        &self,
        prompt: &str,
        parser: P,
    ) -> StructureParserResult<
        Self::TextStream,
        P::Output,
        StructuredTextGenerationError<Self::Error>,
    >
    where
        Self::TextStream: From<tokio::sync::mpsc::UnboundedReceiver<String>>,
        P: CreateParserState<PartialState: Send, Output: Send> + Send + 'static,
    {
        let sampler = Arc::new(Mutex::new(GenerationParameters::default().sampler()));
        let parser_state = parser.create_parser_state();
        self.stream_structured_text_with_sampler(prompt, parser, parser_state, sampler)
    }

    /// Generate structured text with the given prompt and sampler. See [`ModelExt::stream_structured_text`] for more information.
    fn stream_structured_text_with_sampler<P>(
        &self,
        prompt: &str,
        parser: P,
        parser_state: P::PartialState,
        sampler: Arc<Mutex<dyn Sampler>>,
    ) -> StructureParserResult<
        Self::TextStream,
        P::Output,
        StructuredTextGenerationError<Self::Error>,
    >
    where
        Self::TextStream: From<tokio::sync::mpsc::UnboundedReceiver<String>>,
        P: CreateParserState<PartialState: Send, Output: Send> + Send + 'static,
    {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let (result_sender, result_receiver) = tokio::sync::oneshot::channel();

        let prompt = prompt.to_string();
        let result_sender = Arc::new(Mutex::new(Some(result_sender)));
        let result_sender_clone = result_sender.clone();
        if let Err(err) = self.run_sync(move |llm: &mut Self::SyncModel| {
            let mut session = llm
                .new_session()
                .unwrap_or_else(|_| panic!("model must support sessions"));
            Box::pin(async move {
                let result = llm.generate_structured(
                    &mut session,
                    prompt,
                    parser,
                    parser_state,
                    sampler,
                    |token| {
                        _ = sender.send(token);
                        Ok(())
                    },
                    Some(64),
                );
                if let Some(sender) = result_sender.lock().unwrap().take() {
                    _ = sender.send(result);
                }
            })
        }) {
            if let Some(sender) = result_sender_clone.lock().unwrap().take() {
                _ = sender.send(Err(
                    StructuredTextGenerationError::UnstructuredTextGenerationError(
                        UnstructuredTextGenerationError::ModelError(err),
                    ),
                ));
            }
        }

        StructureParserResult::new(Self::TextStream::from(receiver), result_receiver)
    }

    /// Get the default constraints for an assistant response. It parses any text until the end of the assistant's response.
    fn default_assistant_constraints(&self) -> Option<StopOn> {
        let end_assistant_marker = self.chat_markers()?.end_assistant_marker;

        Some(StopOn::from(end_assistant_marker))
    }

    /// Get the constraints that end the assistant's response.
    fn end_assistant_marker_constraints(&self) -> Option<LiteralParser> {
        let end_assistant_marker = self.chat_markers()?.end_assistant_marker;

        Some(LiteralParser::from(end_assistant_marker))
    }
}

/// The result of a structured parser stream.
pub struct StructureParserResult<S: Stream<Item = String> + Send + Unpin + 'static, O, E> {
    stream: S,
    result: tokio::sync::oneshot::Receiver<Result<O, E>>,
}

impl<S: Stream<Item = String> + Send + Unpin + 'static, O, E> StructureParserResult<S, O, E> {
    /// Create a new structured parser result from a stream and a result.
    pub fn new(stream: S, result: tokio::sync::oneshot::Receiver<Result<O, E>>) -> Self {
        Self { stream, result }
    }

    /// Get the final result of the structured parser.
    pub async fn result(self) -> Result<O, E> {
        self.await
    }

    /// Get all the text from the stream.
    pub async fn text(self) -> String {
        let mut text = String::new();
        let mut stream = self.stream;
        while let Some(new) = stream.next().await {
            text.push_str(&new);
        }
        text
    }

    /// Split the stream into a token stream and a result.
    pub fn split(self) -> (S, tokio::sync::oneshot::Receiver<Result<O, E>>) {
        (self.stream, self.result)
    }
}

impl<S: Stream<Item = String> + Send + Unpin + 'static, O, E> Future
    for StructureParserResult<S, O, E>
{
    type Output = Result<O, E>;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let myself = self.get_mut();
        match myself.result.poll_unpin(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            _ => Poll::Pending,
        }
    }
}

impl<S: Stream<Item = String> + Send + Unpin + 'static, O, E> Stream
    for StructureParserResult<S, O, E>
{
    type Item = String;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        this.stream.poll_next_unpin(cx)
    }
}

impl<M: Model + Send + Sync + 'static> ModelExt for M {}
