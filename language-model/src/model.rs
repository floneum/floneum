use crate::embedding::{Embedding, VectorSpace};
use crate::UnknownVectorSpace;
use floneumin_sample::Tokenizer;
use futures_util::{Stream, StreamExt};
use llm_samplers::prelude::Sampler;
use std::future::IntoFuture;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use url::Url;

/// A model that can be used to embed text.
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
#[async_trait::async_trait]
pub trait CreateModel {
    /// Start the model.
    async fn start() -> Self;

    /// Check if the model will need to be downloaded before use (default: false)
    fn requires_download() -> bool {
        false
    }
}

/// A builder for the [`Model::stream_text`] method.
pub struct StreamTextBuilder<'a, M: Model> {
    self_: &'a mut M,
    prompt: &'a str,
    parameters: GenerationParameters,
    future: fn(
        &'a mut M,
        &'a str,
        GenerationParameters,
    ) -> Pin<
        Box<dyn std::future::Future<Output = anyhow::Result<M::TextStream>> + Send + 'a>,
    >,
}

impl<'a, M: Model> StreamTextBuilder<'a, M> {
    /// Create a new builder to return from the [`Model::stream_text`] method.
    pub fn new(
        prompt: &'a str,
        self_: &'a mut M,
        future: fn(
            &'a mut M,
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

    /// Set the top-k to use when generating text.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.parameters.top_k = top_k;
        self
    }

    /// Set the top-p to use when generating text.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.parameters.top_p = top_p;
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

/// A builder for the [`Model::generate_text`] method.
pub struct GenerateTextBuilder<'a, M: Model> {
    self_: &'a mut M,
    prompt: &'a str,
    parameters: GenerationParameters,
    future: fn(
        &'a mut M,
        &'a str,
        GenerationParameters,
    )
        -> Pin<Box<dyn std::future::Future<Output = anyhow::Result<String>> + Send + 'a>>,
}

impl<'a, M: Model> GenerateTextBuilder<'a, M> {
    /// Create a new builder to return from the [`Model::generate_text`] method.
    pub fn new(
        prompt: &'a str,
        self_: &'a mut M,
        future: fn(
            &'a mut M,
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

    /// Set the top-k to use when generating text.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.parameters.top_k = top_k;
        self
    }

    /// Set the top-p to use when generating text.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.parameters.top_p = top_p;
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

/// A model that can be used to generate text with an associated tokenizer.
///
/// The model may support using a custom sampler. If a specific model does not support a specific method, it will return an error.
#[async_trait::async_trait]
pub trait Model: Send + 'static {
    /// The type of stream that this model generates.
    type TextStream: Stream<Item = String> + Send + Sync + Unpin + 'static;

    /// Get the tokenizer associated with this model to use for constrained generation.
    fn tokenizer(&self) -> Arc<dyn Tokenizer + Send + Sync>;

    /// Generate text with the given prompt.
    async fn generate_text_with_sampler(
        &mut self,
        prompt: &str,
        max_tokens: Option<u32>,
        stop_on: Option<&str>,
        sampler: Arc<Mutex<dyn Sampler<u32, f32>>>,
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
    fn generate_text<'a>(&'a mut self, prompt: &'a str) -> GenerateTextBuilder<'a, Self>
    where
        Self: Sized + Send + Sync,
    {
        GenerateTextBuilder::new(prompt, self, |self_, prompt, generation_parameters| {
            Box::pin(async {
                let mut text = String::new();

                let mut stream = self_
                    .stream_text(prompt)
                    .with_generation_parameters(generation_parameters)
                    .await?;
                while let Some(new) = stream.next().await {
                    text.push_str(&new);
                }
                Ok(text)
            })
        })
    }

    /// Generate text with the given prompt.
    async fn stream_text_with_sampler(
        &mut self,
        _prompt: &str,
        _max_tokens: Option<u32>,
        _stop_on: Option<&str>,
        _sampler: Arc<Mutex<dyn Sampler<u32, f32>>>,
    ) -> anyhow::Result<Self::TextStream> {
        Err(anyhow::Error::msg("Not implemented"))
    }

    /// Generate text with the given prompt.
    fn stream_text<'a>(&'a mut self, prompt: &'a str) -> StreamTextBuilder<'a, Self>
    where
        Self: Sized;

    /// Convert this model into a model trait object.
    fn into_any_model(self) -> DynModel
    where
        Self: Send + Sync + Sized,
    {
        Box::new(AnyModel(self, PhantomData))
    }
}

/// A trait object for a model.
pub type DynModel =
    Box<dyn Model<TextStream = Box<dyn Stream<Item = String> + Send + Sync + Unpin>> + Send + Sync>;

struct AnyModel<
    M: Model<TextStream = S> + Send + Sync,
    S: Stream<Item = String> + Send + Sync + Unpin + 'static,
>(M, PhantomData<S>);

#[async_trait::async_trait]
impl<M, S> Model for AnyModel<M, S>
where
    S: Stream<Item = String> + Send + Sync + Unpin + 'static,
    M: Model<TextStream = S> + Send + Sync,
{
    type TextStream = Box<dyn Stream<Item = String> + Send + Sync + Unpin>;

    fn tokenizer(&self) -> Arc<dyn Tokenizer + Send + Sync> {
        self.0.tokenizer()
    }

    fn stream_text<'a>(&'a mut self, prompt: &'a str) -> StreamTextBuilder<'a, Self> {
        StreamTextBuilder::new(prompt, self, |self_, prompt, generation_parameters| {
            Box::pin(async {
                self_
                    .0
                    .stream_text(prompt)
                    .with_generation_parameters(generation_parameters)
                    .await
                    .map(|s| Box::new(s) as Box<dyn Stream<Item = String> + Send + Sync + Unpin>)
            })
        })
    }

    async fn stream_text_with_sampler(
        &mut self,
        prompt: &str,
        max_tokens: Option<u32>,
        stop_on: Option<&str>,
        sampler: Arc<Mutex<dyn Sampler<u32, f32>>>,
    ) -> anyhow::Result<Self::TextStream> {
        self.0
            .stream_text_with_sampler(prompt, max_tokens, stop_on, sampler)
            .await
            .map(|s| Box::new(s) as Box<dyn Stream<Item = String> + Send + Sync + Unpin>)
    }
}

/// Parameters to use when generating text.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationParameters {
    pub(crate) temperature: f32,
    pub(crate) top_k: u32,
    pub(crate) top_p: f32,
    pub(crate) repetition_penalty: f32,
    pub(crate) repetition_penalty_range: u32,
    pub(crate) max_length: u32,
    pub(crate) stop_on: Option<String>,
}

impl Default for GenerationParameters {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.3,
            repetition_penalty_range: 64,
            max_length: 128,
            stop_on: None,
        }
    }
}

impl GenerationParameters {
    /// Create a gready generation configuration that will always return the most likely token.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.8,
            top_k: 1,
            top_p: 0.95,
            repetition_penalty: 1.3,
            repetition_penalty_range: 64,
            max_length: 128,
            stop_on: None,
        }
    }

    /// Set the temperature to use when generating text.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the top-k to use when generating text.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the top-p to use when generating text.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
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

    /// Get the top-k to use when generating text.
    pub fn top_k(&self) -> u32 {
        self.top_k
    }

    /// Get the top-p to use when generating text.
    pub fn top_p(&self) -> f32 {
        self.top_p
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
