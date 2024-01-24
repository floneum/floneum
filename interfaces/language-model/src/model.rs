use crate::embedding::{Embedding, VectorSpace};
use crate::structured::generate_structured;
use crate::TokenOutputStream;
use crate::UnknownVectorSpace;
use futures_util::{Stream, StreamExt};
use kalosm_sample::{Parser, Tokenizer};
use kalosm_streams::text_stream::ChannelTextStream;
use llm_samplers::configure::SamplerChainBuilder;
use llm_samplers::prelude::*;
use llm_samplers::types::Logits;
use std::any::Any;
use std::fmt::Display;
use std::future::IntoFuture;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use url::Url;

/// A model that can be used to embed text. This trait is generic over the vector space that the model uses to help keep track of what embeddings came from which model.
///
/// # Example
///
/// ```rust, no_run
/// use kalosm_language_model::Embedder;
/// use rbert::*;
///
/// #[tokio::main]
/// async fn main() {
///     // Bert implements Embedder
///     let mut bert = Bert::builder().build().unwrap();
///     let sentences = vec![
///         "Cats are cool",
///         "The geopolitical situation is dire",
///         "Pets are great",
///         "Napoleon was a tyrant",
///         "Napoleon was a great general",
///     ];
///     // Embed a batch of documents into the bert vector space
///     let embeddings = bert.embed_batch(&sentences).await.unwrap();
///     println!("embeddings {:?}", embeddings);
/// }
/// ```
#[async_trait::async_trait]
pub trait Embedder<S: VectorSpace + Send + Sync + 'static>: Send + Sync + 'static {
    /// Embed a single string.
    async fn embed(&mut self, input: &str) -> anyhow::Result<Embedding<S>>;

    /// Embed a batch of strings.
    async fn embed_batch(&mut self, inputs: &[&str]) -> anyhow::Result<Vec<Embedding<S>>> {
        let mut embeddings = Vec::with_capacity(inputs.len());
        for input in inputs {
            embeddings.push(self.embed(input).await?);
        }
        Ok(embeddings)
    }

    /// Convert this embedder into an embedder trait object.
    fn into_any_embedder(self) -> DynEmbedder
    where
        Self: Sized,
    {
        Box::new(AnyEmbedder::<S, Self>(self, PhantomData))
    }
}

/// A trait object for an embedder.
pub type DynEmbedder = Box<dyn Embedder<UnknownVectorSpace>>;

struct AnyEmbedder<S: VectorSpace + Send + Sync + 'static, E: Embedder<S> + Send + Sync + 'static>(
    E,
    PhantomData<S>,
);

#[async_trait::async_trait]
impl<S: VectorSpace + Send + Sync + 'static, E: Embedder<S> + Send + Sync + 'static>
    Embedder<UnknownVectorSpace> for AnyEmbedder<S, E>
{
    async fn embed(&mut self, input: &str) -> anyhow::Result<Embedding<UnknownVectorSpace>> {
        self.0.embed(input).await.map(|e| e.cast())
    }

    async fn embed_batch(
        &mut self,
        inputs: &[&str],
    ) -> anyhow::Result<Vec<Embedding<UnknownVectorSpace>>> {
        self.0
            .embed_batch(inputs)
            .await
            .map(|e| e.into_iter().map(|e| e.cast()).collect())
    }
}

/// A model that can be created asynchronously.
///
/// # Example
/// ```rust, no_run
/// use rbert::*;
/// use kalosm_language_model::CreateModel;
///
/// #[tokio::main]
/// async fn main() {
///     let mut bert = Bert::start().await;
/// }
/// ```
#[async_trait::async_trait]
pub trait CreateModel {
    /// Start the model.
    async fn start() -> Self;

    /// Check if the model will need to be downloaded before use (default: false)
    fn requires_download() -> bool {
        false
    }
}

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
        Box<dyn std::future::Future<Output = anyhow::Result<M::TextStream>> + Send + 'a>,
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
            Box<dyn std::future::Future<Output = anyhow::Result<M::TextStream>> + Send + 'a>,
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
    type Output = anyhow::Result<M::TextStream>;
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
        -> Pin<Box<dyn std::future::Future<Output = anyhow::Result<String>> + Send + 'a>>,
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
            Box<dyn std::future::Future<Output = anyhow::Result<String>> + Send + 'a>,
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
    type Output = anyhow::Result<String>;
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

/// An extension trait for models.
#[async_trait::async_trait]
pub trait ModelExt: Model + Send + Sync + 'static {
    /// Generate text with the given prompt. This function generates a builder with extra parameters that can be set. To execute the builder, just call `await` on it.
    ///
    /// ```rust, no_run
    /// use rphi::prelude::*;
    /// use std::io::Write;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let mut model = Phi::default();
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
    /// use rphi::prelude::*;
    /// use std::io::Write;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let mut model = Phi::default();
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
    /// use rphi::prelude::*;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let mut llm = Phi::start().await;
    ///
    ///     let tokenizer = llm.tokenizer();
    ///     // Start a sync task on the model
    ///     llm.run_sync(move |llm: &mut <Phi as Model>::SyncModel| {
    ///         Box::pin(async move {
    ///             let question = "What is 10 + 10?";
    ///
    ///             // Create a new session of the model
    ///             let mut session = llm.new_session().unwrap();
    ///
    ///             // Feed the question into the model
    ///             let mut logits = llm.feed_text(&mut session, question).unwrap();
    ///
    ///             println!("logits: {:?}", logits);
    ///         })
    ///     })
    ///     .await
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
    ) -> anyhow::Result<()> {
        self.run_sync_raw(Box::new(f))
    }

    /// Generate structured text with the given prompt.
    async fn stream_structured_text<P>(
        &self,
        prompt: &str,
        parser: P,
    ) -> anyhow::Result<StructureParserResult<Self::TextStream, P::Output>>
    where
        Self::TextStream: From<tokio::sync::mpsc::UnboundedReceiver<String>>,
        P: kalosm_sample::CreateParserState + Parser + Send + 'static,
        P::PartialState: Send + 'static,
        P::Output: Clone + Send + 'static,
    {
        let sampler = Arc::new(Mutex::new(GenerationParameters::default().sampler()));
        let parser_state = parser.create_parser_state();
        self.stream_structured_text_with_sampler(prompt, parser, parser_state, sampler)
            .await
    }

    /// Generate structured text with the given prompt and sampler.
    async fn stream_structured_text_with_sampler<P>(
        &self,
        prompt: &str,
        parser: P,
        parser_state: P::PartialState,
        sampler: Arc<Mutex<dyn Sampler>>,
    ) -> anyhow::Result<StructureParserResult<Self::TextStream, P::Output>>
    where
        Self::TextStream: From<tokio::sync::mpsc::UnboundedReceiver<String>>,
        P: Parser + Send + 'static,
        P::PartialState: Send + 'static,
        P::Output: Clone + Send + 'static,
    {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let (result_sender, result_receiver) = tokio::sync::oneshot::channel();

        let prompt = prompt.to_string();
        self.run_sync(move |llm: &mut Self::SyncModel| {
            let mut session = llm.new_session().unwrap();
            Box::pin(async move {
                let result = llm.generate_structured(
                    &mut session,
                    prompt,
                    parser,
                    parser_state,
                    sampler,
                    |token| Ok(sender.send(token)?),
                );
                match result_sender.send(result) {
                    Ok(()) => {}
                    Err(Ok(_)) => {
                        log::error!("Error generating structured text: cancelled");
                    }
                    Err(Err(err)) => {
                        log::error!("Error generating structured text: {:?}", err);
                    }
                }
            })
        })?;

        Ok(StructureParserResult::new(
            Self::TextStream::from(receiver),
            result_receiver,
        ))
    }
}

/// The result of a structured parser stream.
pub struct StructureParserResult<S: Stream<Item = String> + Send + Unpin + 'static, O> {
    stream: S,
    result: tokio::sync::oneshot::Receiver<anyhow::Result<O>>,
}

impl<S: Stream<Item = String> + Send + Unpin + 'static, O> Deref for StructureParserResult<S, O> {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.stream
    }
}

impl<S: Stream<Item = String> + Send + Unpin + 'static, O> DerefMut
    for StructureParserResult<S, O>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.stream
    }
}

impl<S: Stream<Item = String> + Send + Unpin + 'static, O> StructureParserResult<S, O> {
    /// Create a new structured parser result from a stream and a result.
    pub fn new(stream: S, result: tokio::sync::oneshot::Receiver<anyhow::Result<O>>) -> Self {
        Self { stream, result }
    }

    /// Get the final result of the structured parser.
    pub async fn result(self) -> anyhow::Result<O> {
        self.result.await.unwrap()
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
    pub fn split(self) -> (S, tokio::sync::oneshot::Receiver<anyhow::Result<O>>) {
        (self.stream, self.result)
    }
}

impl<M: Model + Send + Sync + 'static> ModelExt for M {}

/// A raw interface for a model that can be used to generate text synchronously. This provides a very low level interface to a model's session:
///
/// # Example
/// ```rust, no_run
/// use rphi::prelude::*;
///
/// #[tokio::main]
/// async fn main() {
///     let mut llm = Phi::start().await;
///
///     let tokenizer = llm.tokenizer();
///     // Start a sync task on the model
///     llm.run_sync(move |llm: &mut <Phi as Model>::SyncModel| {
///         Box::pin(async move {
///             let question = "What is 10 + 10?";
///
///             // Create a new session of the model
///             let mut session = llm.new_session().unwrap();
///
///             // Feed the question into the model
///             let mut logits = llm.feed_text(&mut session, question).unwrap();
///
///             println!("logits: {:?}", logits);
///         })
///     })
///     .await
///     .unwrap();
/// }
/// ```
pub trait SyncModel {
    /// The session type for this model.
    type Session: Session;

    /// Create a new session for this model.
    fn new_session(&self) -> anyhow::Result<Self::Session>;

    /// Run the model synchronously. The model implementation may choose to return only the top k logits.
    fn feed_text(
        &self,
        session: &mut Self::Session,
        prompt: &str,
        top_k: Option<usize>,
    ) -> anyhow::Result<Logits>;

    /// Run the model synchronously with a pre-tokenized input. The model implementation may choose to return only the top k logits.
    fn feed_tokens(
        &self,
        session: &mut Self::Session,
        tokens: &[u32],
        top_k: Option<usize>,
    ) -> anyhow::Result<Logits>;

    /// Get the token ID that represents the end of a sequence.
    fn stop_token(&self) -> anyhow::Result<u32>;

    /// Return the tokenizer associated with this model.
    fn tokenizer(&self) -> Arc<dyn Tokenizer + Send + Sync>;
}

/// A session for a model.
pub trait Session {
    /// Save the session to the given path.
    fn save_to(&self, _path: impl AsRef<Path>) -> anyhow::Result<()> {
        Err(anyhow::Error::msg("Not implemented"))
    }

    /// Load the session from the given path.
    fn load_from(_path: impl AsRef<Path>) -> anyhow::Result<Self>
    where
        Self: std::marker::Sized,
    {
        Err(anyhow::Error::msg("Not implemented"))
    }

    /// Try to clone the session.
    fn try_clone(&self) -> anyhow::Result<Self>
    where
        Self: std::marker::Sized,
    {
        Err(anyhow::Error::msg("Not implemented"))
    }
}

impl Session for () {
    fn save_to(&self, _path: impl AsRef<Path>) -> anyhow::Result<()> {
        Ok(())
    }

    fn load_from(_path: impl AsRef<Path>) -> anyhow::Result<()> {
        Ok(())
    }
}

/// An extension trait for sync models.
pub trait SyncModelExt: SyncModel {
    /// Generate new text with the given prompt that conforms to the given parser.
    fn generate_structured<P: Parser>(
        &self,
        session: &mut Self::Session,
        prompt: impl Display,
        parser: P,
        parser_state: P::PartialState,
        sampler: Arc<Mutex<dyn Sampler>>,
        on_token: impl FnMut(String) -> anyhow::Result<()>,
    ) -> anyhow::Result<P::Output>
    where
        P::Output: Clone,
    {
        let tokenizer = self.tokenizer();
        let stop_token = self
            .stop_token()
            .ok()
            .and_then(|token_id| tokenizer.decode(&[token_id]).ok())
            .map(|s| s.to_string())
            .unwrap_or("<|endoftext|>".to_string());
        generate_structured(
            prompt,
            self,
            session,
            &tokenizer,
            stop_token,
            parser,
            parser_state,
            sampler,
            on_token,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Stream text, calling the on_token callback every time a new token is generated. For some models, this could be used to implement [`Model::stream_text_with_sampler`].
    fn stream_text_with_sampler(
        &self,
        session: &mut Self::Session,
        prompt: &str,
        max_tokens: Option<u32>,
        stop_on: Option<&str>,
        mut sampler: Arc<Mutex<dyn Sampler>>,
        mut on_token: impl FnMut(String) -> anyhow::Result<ModelFeedback>,
    ) -> anyhow::Result<()> {
        let tokens = self.tokenizer().encode(prompt)?;
        let mut text_stream = TokenOutputStream::new(self.tokenizer(), tokens.clone());

        let mut logits = self.feed_tokens(session, &tokens, Some(512))?;
        let mut tokens_generated = 0;
        // This stores a buffer of text that has been generated to check against the stop_on string. It should never be longer than the stop_on string.
        let mut queued_text_matching_stop_on = String::new();
        let stop_on_lowercase = stop_on.map(|s| s.to_lowercase());
        let stop_on_lowercase = stop_on_lowercase.as_deref();
        let stop_token = self.stop_token()?;

        'generate: loop {
            let new_token = text_stream.sample_token(&mut sampler, logits, stop_on)?;
            if new_token == stop_token {
                break;
            }
            if let Some(mut new_text) = text_stream.next_token(new_token)? {
                if let Some(stop_on) = stop_on_lowercase {
                    let lowercase = new_text.to_lowercase();

                    // Check if the string ends with the start of the stop_on string
                    let mut before_stop_on = None;
                    let remaining_stop_on = stop_on
                        .strip_prefix(&queued_text_matching_stop_on)
                        .unwrap_or(stop_on);

                    // If the remaining stop_on string is empty, we have found a match
                    if remaining_stop_on.is_empty() {
                        break;
                    }

                    for (i, _) in lowercase.char_indices() {
                        let end_of_new_text = &lowercase[i..];
                        if end_of_new_text.is_empty() {
                            break;
                        }

                        // Check if we have matched all of the stop_on string
                        if end_of_new_text.starts_with(remaining_stop_on) {
                            queued_text_matching_stop_on += end_of_new_text;
                            break 'generate;
                        }

                        // Check if the string ends with the start of the stop_on string
                        if remaining_stop_on.starts_with(end_of_new_text) {
                            before_stop_on = Some(lowercase[..i].to_string());
                            queued_text_matching_stop_on += end_of_new_text;
                            break;
                        }
                    }

                    match before_stop_on {
                        Some(before_stop_on) => {
                            if let ModelFeedback::Stop = on_token(before_stop_on)? {
                                break;
                            }
                        }
                        None => {
                            new_text =
                                std::mem::take(&mut queued_text_matching_stop_on) + &new_text;
                            if let ModelFeedback::Stop = on_token(new_text)? {
                                break;
                            }
                        }
                    }
                } else if let ModelFeedback::Stop = on_token(new_text)? {
                    break;
                }
            }
            tokens_generated += 1;
            if let Some(max_tokens) = max_tokens {
                if tokens_generated >= max_tokens {
                    break;
                }
            }
            logits = self.feed_tokens(session, &[new_token], Some(512))?;
        }

        // Flush the queued text
        if let Some(stop_string) = stop_on_lowercase {
            if !queued_text_matching_stop_on.starts_with(stop_string) {
                on_token(queued_text_matching_stop_on)?;
            }
        }

        Ok(())
    }
}

/// Feedback to give to the model when generating text.
pub enum ModelFeedback {
    /// Continue generating text.
    Continue,
    /// Stop generating text.
    Stop,
}

impl<M: SyncModel> SyncModelExt for M {}

/// A marker type for models that do not support synchronous generation.
pub struct SyncModelNotSupported;

impl SyncModel for SyncModelNotSupported {
    type Session = ();

    fn new_session(&self) -> anyhow::Result<Self::Session> {
        Err(anyhow::Error::msg("Not implemented"))
    }

    fn feed_text(
        &self,
        _session: &mut (),
        _prompt: &str,
        _: Option<usize>,
    ) -> anyhow::Result<Logits> {
        Err(anyhow::Error::msg("Not implemented"))
    }

    fn feed_tokens(
        &self,
        _session: &mut (),
        _tokens: &[u32],
        _: Option<usize>,
    ) -> anyhow::Result<Logits> {
        Err(anyhow::Error::msg("Not implemented"))
    }

    fn stop_token(&self) -> anyhow::Result<u32> {
        Err(anyhow::Error::msg("Not implemented"))
    }

    fn tokenizer(&self) -> Arc<dyn Tokenizer + Send + Sync> {
        unimplemented!()
    }
}

/// A model that can be used to generate text with an associated tokenizer.
///
/// The model may support using a custom sampler. If a specific model does not support a specific method, it will return an error.
#[async_trait::async_trait]
pub trait Model: Send + Sync + 'static {
    /// The type of stream that this model generates.
    type TextStream: Stream<Item = String> + Send + Sync + Unpin + 'static;

    /// Get the tokenizer associated with this model to use for constrained generation.
    fn tokenizer(&self) -> Arc<dyn Tokenizer + Send + Sync>;

    /// The raw sync model that backs this model.
    type SyncModel: SyncModel;

    #[allow(clippy::type_complexity)]
    /// Run some code synchronously with the model.
    ///
    /// See [`ModelExt::run_sync`] for nicer API with an example.
    fn run_sync_raw(
        &self,
        _f: Box<
            dyn for<'a> FnOnce(
                    &'a mut Self::SyncModel,
                )
                    -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>>
                + Send,
        >,
    ) -> anyhow::Result<()> {
        Err(anyhow::Error::msg("Not implemented"))
    }

    /// Generate text with the given prompt.
    async fn generate_text_with_sampler(
        &self,
        prompt: &str,
        max_tokens: Option<u32>,
        stop_on: Option<&str>,
        sampler: Arc<Mutex<dyn Sampler>>,
    ) -> anyhow::Result<String> {
        let mut text = String::new();

        let mut stream = self
            .stream_text_with_sampler(prompt, max_tokens, stop_on, sampler)
            .await?;
        while let Some(new) = stream.next().await {
            text.push_str(&new);
        }
        Ok(text)
    }

    /// Generate text with the given prompt.
    ///
    /// See [`ModelExt::generate_text`] for nicer API with an example.
    async fn generate_text_inner(
        &self,
        prompt: &str,
        parameters: GenerationParameters,
    ) -> anyhow::Result<String> {
        let mut text = String::new();

        let mut stream = self.stream_text_inner(prompt, parameters).await?;
        while let Some(new) = stream.next().await {
            text.push_str(&new);
        }
        Ok(text)
    }

    /// Generate text with the given prompt.
    async fn stream_text_with_sampler(
        &self,
        _prompt: &str,
        _max_tokens: Option<u32>,
        _stop_on: Option<&str>,
        _sampler: Arc<Mutex<dyn Sampler>>,
    ) -> anyhow::Result<Self::TextStream> {
        Err(anyhow::Error::msg("Not implemented"))
    }

    /// Generate text with the given prompt.
    ///
    /// See [`ModelExt::stream_text`] for nicer API with an example.
    async fn stream_text_inner(
        &self,
        prompt: &str,
        parameters: GenerationParameters,
    ) -> anyhow::Result<Self::TextStream>;

    /// Returns the chat markers to use for the model if this is a chat model.
    fn chat_markers(&self) -> Option<ChatMarkers> {
        None
    }
}

/// An extension trait for models that can be converted into a trait object.
pub trait AnyModelExt:
    Model<TextStream = ChannelTextStream<String>> + Send + Sync + 'static
{
    /// Convert this model into a model trait object.
    fn into_any_model(self) -> DynModel
    where
        Self: Send + Sync + Sized,
    {
        Box::new(AnyModel(self))
    }
}

impl<M: Model<TextStream = ChannelTextStream<String>> + Send + Sync + 'static> AnyModelExt for M {}

/// The chat markers to use for the model.
#[derive(Default, Clone)]
pub struct ChatMarkers {
    /// The marker to use before user input.
    pub user_marker: &'static str,
    /// The marker to use after user input.
    pub end_user_marker: &'static str,
    /// The marker to use before assistant messages.
    pub assistant_marker: &'static str,
    /// The marker to use after assistant messages.
    pub end_assistant_marker: &'static str,
    /// The marker to use before system prompts.
    pub system_prompt_marker: &'static str,
    /// The marker to use after system prompts.
    pub end_system_prompt_marker: &'static str,
}

/// A trait object for a model.
pub type DynModel =
    Box<dyn Model<TextStream = ChannelTextStream<String>, SyncModel = BoxedSyncModel> + Send>;

#[async_trait::async_trait]
impl Model for DynModel {
    type TextStream = ChannelTextStream<String>;
    type SyncModel = BoxedSyncModel;

    fn tokenizer(&self) -> Arc<dyn Tokenizer + Send + Sync> {
        let self_ref: &(dyn Model<TextStream = ChannelTextStream<String>, SyncModel = BoxedSyncModel>
              + Send) = self.as_ref();
        self_ref.tokenizer()
    }

    async fn stream_text_inner(
        &self,
        prompt: &str,
        parameters: GenerationParameters,
    ) -> anyhow::Result<Self::TextStream> {
        let self_ref: &(dyn Model<TextStream = ChannelTextStream<String>, SyncModel = BoxedSyncModel>
              + Send) = self.as_ref();
        self_ref.stream_text_inner(prompt, parameters).await
    }
}

/// A trait object for a sync model.
pub type BoxedSyncModel = Box<dyn SyncModel<Session = AnySession>>;

trait AnySessionTrait {
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn save_to(&self, path: &Path) -> anyhow::Result<()>;
}

impl<S: Any + Session> AnySessionTrait for S {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn save_to(&self, path: &Path) -> anyhow::Result<()> {
        Session::save_to(self, path)
    }
}

/// A type-erased session.
///
/// > Note: boxed sessions do not support loading from a path.
pub struct AnySession {
    session: Box<dyn AnySessionTrait>,
}

impl AnySession {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self.session.as_any_mut()
    }
}

impl Session for AnySession {
    fn save_to(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        self.session.save_to(path.as_ref())
    }
}

impl SyncModel for BoxedSyncModel {
    type Session = AnySession;

    fn new_session(&self) -> anyhow::Result<Self::Session> {
        let self_ref: &(dyn SyncModel<Session = AnySession>) = self.as_ref();
        self_ref.new_session()
    }

    fn feed_text(
        &self,
        session: &mut Self::Session,
        prompt: &str,
        top_k: Option<usize>,
    ) -> anyhow::Result<Logits> {
        let self_ref: &(dyn SyncModel<Session = AnySession>) = self.as_ref();
        self_ref.feed_text(session, prompt, top_k)
    }

    fn feed_tokens(
        &self,
        session: &mut Self::Session,
        tokens: &[u32],
        top_k: Option<usize>,
    ) -> anyhow::Result<Logits> {
        let self_ref: &(dyn SyncModel<Session = AnySession>) = self.as_ref();
        self_ref.feed_tokens(session, tokens, top_k)
    }

    fn stop_token(&self) -> anyhow::Result<u32> {
        let self_ref: &(dyn SyncModel<Session = AnySession>) = self.as_ref();
        self_ref.stop_token()
    }

    fn tokenizer(&self) -> Arc<dyn Tokenizer + Send + Sync> {
        let self_ref: &(dyn SyncModel<Session = AnySession>) = self.as_ref();
        self_ref.tokenizer()
    }
}

struct AnySyncModel<M: SyncModel<Session = S>, S: Any>(M, PhantomData<S>);

impl<M: SyncModel<Session = S>, S: Session + Any> SyncModel for AnySyncModel<M, S> {
    type Session = AnySession;

    fn new_session(&self) -> anyhow::Result<Self::Session> {
        self.0.new_session().map(|s| AnySession {
            session: Box::new(s),
        })
    }

    fn feed_text(
        &self,
        session: &mut Self::Session,
        prompt: &str,
        top_k: Option<usize>,
    ) -> anyhow::Result<Logits> {
        self.0.feed_text(
            match session.as_any_mut().downcast_mut() {
                Some(s) => s,
                None => {
                    return Err(anyhow::Error::msg(format!(
                        "Invalid session type expected {:?}",
                        std::any::type_name::<S>()
                    )))
                }
            },
            prompt,
            top_k,
        )
    }

    fn feed_tokens(
        &self,
        session: &mut Self::Session,
        tokens: &[u32],
        top_k: Option<usize>,
    ) -> anyhow::Result<Logits> {
        self.0.feed_tokens(
            match session.as_any_mut().downcast_mut() {
                Some(s) => s,
                None => {
                    return Err(anyhow::Error::msg(format!(
                        "Invalid session type expected {:?}",
                        std::any::type_name::<S>()
                    )))
                }
            },
            tokens,
            top_k,
        )
    }

    fn stop_token(&self) -> anyhow::Result<u32> {
        self.0.stop_token()
    }

    fn tokenizer(&self) -> Arc<dyn Tokenizer + Send + Sync> {
        self.0.tokenizer()
    }
}

struct AnyModel<M>(M);

#[async_trait::async_trait]
impl<M> Model for AnyModel<M>
where
    M: Model<TextStream = ChannelTextStream<String>> + Send + Sync,
{
    type TextStream = ChannelTextStream<String>;
    type SyncModel = BoxedSyncModel;

    fn tokenizer(&self) -> Arc<dyn Tokenizer + Send + Sync> {
        self.0.tokenizer()
    }

    async fn stream_text_inner(
        &self,
        prompt: &str,
        params: GenerationParameters,
    ) -> anyhow::Result<Self::TextStream> {
        self.0.stream_text_inner(prompt, params).await
    }

    async fn stream_text_with_sampler(
        &self,
        prompt: &str,
        max_tokens: Option<u32>,
        stop_on: Option<&str>,
        sampler: Arc<Mutex<dyn Sampler>>,
    ) -> anyhow::Result<Self::TextStream> {
        self.0
            .stream_text_with_sampler(prompt, max_tokens, stop_on, sampler)
            .await
    }
}

/// Parameters to use when generating text.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationParameters {
    pub(crate) temperature: f32,
    pub(crate) tau: f32,
    pub(crate) eta: f32,
    pub(crate) mu: f32,
    pub(crate) repetition_penalty: f32,
    pub(crate) repetition_penalty_range: u32,
    pub(crate) max_length: u32,
    pub(crate) stop_on: Option<String>,
}

impl Default for GenerationParameters {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            eta: 0.1,
            tau: 5.,
            mu: 10.,
            repetition_penalty: 1.3,
            repetition_penalty_range: 64,
            max_length: 128,
            stop_on: None,
        }
    }
}

impl crate::model::GenerationParameters {
    /// Create a sampler chain from the generation parameters.
    pub fn sampler(self) -> SamplerChain {
        use llm_samplers::configure::SamplerSlot;
        let GenerationParameters {
            temperature,
            tau,
            eta,
            mu,
            repetition_penalty,
            repetition_penalty_range,
            max_length: _,
            stop_on: _,
        } = self;
        SamplerChainBuilder::from([
            (
                "repetition",
                SamplerSlot::new_static(move || {
                    Box::new(
                        SampleRepetition::default()
                            .penalty(repetition_penalty)
                            .last_n(repetition_penalty_range as usize),
                    )
                }),
            ),
            (
                "freqpresence",
                SamplerSlot::new_static(move || Box::new(SampleFreqPresence::default().last_n(64))),
            ),
            (
                "seqrepetition",
                SamplerSlot::new_static(move || Box::<SampleSeqRepetition>::default()),
            ),
            (
                "temperature",
                SamplerSlot::new_static(move || {
                    Box::new(SampleTemperature::default().temperature(temperature))
                }),
            ),
            (
                "mirostat2",
                SamplerSlot::new_static(move || {
                    Box::new(SampleMirostat2::default().tau(tau).eta(eta).mu(mu))
                }),
            ),
        ])
        .into_chain()
    }

    /// Get the mirostat2 sampler from the generation parameters.
    pub fn mirostat2_sampler(self) -> SampleMirostat2 {
        SampleMirostat2::default()
            .tau(self.tau)
            .eta(self.eta)
            .mu(self.mu)
    }

    /// Create a sampler chain from the generation parameters without removing any tokens. This can be useful in combination with [`ModelExt::stream_structured_text_with_sampler`] which may pick unlikely tokens.
    pub fn bias_only_sampler(self) -> SamplerChain {
        use llm_samplers::configure::SamplerSlot;
        let GenerationParameters {
            temperature,
            repetition_penalty,
            repetition_penalty_range,
            ..
        } = self;
        SamplerChainBuilder::from([
            (
                "repetition",
                SamplerSlot::new_static(move || {
                    Box::new(
                        SampleRepetition::default()
                            .penalty(repetition_penalty)
                            .last_n(repetition_penalty_range as usize),
                    )
                }),
            ),
            (
                "freqpresence",
                SamplerSlot::new_static(move || Box::new(SampleFreqPresence::default().last_n(64))),
            ),
            (
                "seqrepetition",
                SamplerSlot::new_static(move || Box::<SampleSeqRepetition>::default()),
            ),
            (
                "temperature",
                SamplerSlot::new_static(move || {
                    Box::new(SampleTemperature::default().temperature(temperature))
                }),
            ),
        ])
        .into_chain()
    }

    /// Set the temperature to use when generating text.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the tau to use when generating text.
    pub fn with_tau(mut self, tau: f32) -> Self {
        self.tau = tau;
        self
    }

    /// Set the eta to use when generating text.
    pub fn with_eta(mut self, eta: f32) -> Self {
        self.eta = eta;
        self
    }

    /// Set the mu to use when generating text.
    pub fn with_mu(mut self, mu: f32) -> Self {
        self.mu = mu;
        self
    }

    /// Set the repetition penalty to use when generating text.
    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = repetition_penalty;
        self
    }

    /// Set the repetition penalty range to use when generating text.
    pub fn with_repetition_penalty_range(mut self, repetition_penalty_range: u32) -> Self {
        self.repetition_penalty_range = repetition_penalty_range;
        self
    }

    /// Set the maximum length to use when generating text.
    pub fn with_max_length(mut self, max_length: u32) -> Self {
        self.max_length = max_length;
        self
    }

    /// Set the string to stop on when generating text.
    pub fn with_stop_on(mut self, stop_on: impl Into<Option<String>>) -> Self {
        self.stop_on = stop_on.into();
        self
    }

    /// Get the temperature to use when generating text.
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Get the tau to use when generating text.
    pub fn tau(&self) -> f32 {
        self.tau
    }

    /// Get the eta to use when generating text.
    pub fn eta(&self) -> f32 {
        self.eta
    }

    /// Get the mu to use when generating text.
    pub fn mu(&self) -> f32 {
        self.mu
    }

    /// Get the repetition penalty to use when generating text.
    pub fn repetition_penalty(&self) -> f32 {
        self.repetition_penalty
    }

    /// Get the repetition penalty range to use when generating text.
    pub fn repetition_penalty_range(&self) -> u32 {
        self.repetition_penalty_range
    }

    /// Get the maximum length to use when generating text.
    pub fn max_length(&self) -> u32 {
        self.max_length
    }

    /// Get the string to stop on when generating text.
    pub fn stop_on(&self) -> Option<&str> {
        self.stop_on.as_deref()
    }
}

/// The type of model to use.
#[allow(missing_docs)]
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ModelType {
    Mpt(MptType),
    GptNeoX(GptNeoXType),
    Llama(LlamaType),
}

/// The type of Llama model to use.
#[allow(missing_docs)]
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum LlamaType {
    Vicuna,
    Guanaco,
    WizardLm,
    Orca,
    LlamaSevenChat,
    LlamaThirteenChat,
    Custom(Url),
}

/// The type of MPT model to use.
#[allow(missing_docs)]
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum MptType {
    Base,
    Story,
    Instruct,
    Chat,
    Custom(Url),
}

/// The type of GPT-NeoX model to use.
#[allow(missing_docs)]
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum GptNeoXType {
    LargePythia,
    TinyPythia,
    DollySevenB,
    StableLm,
    Custom(Url),
}

macro_rules! embedding {
    ($ty: ident) => {
        #[doc = "A vector space for the "]
        #[doc = stringify!($ty)]
        #[doc = " model."]
        pub struct $ty;

        impl VectorSpace for $ty {}
    };
}

embedding!(VicunaSpace);
embedding!(GuanacoSpace);
embedding!(WizardLmSpace);
embedding!(OrcaSpace);
embedding!(LlamaSevenChatSpace);
embedding!(LlamaThirteenChatSpace);
embedding!(MptBaseSpace);
embedding!(MptStorySpace);
embedding!(MptInstructSpace);
embedding!(MptChatSpace);
embedding!(LargePythiaSpace);
embedding!(TinyPythiaSpace);
embedding!(DollySevenBSpace);
embedding!(StableLmSpace);
