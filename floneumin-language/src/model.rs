use crate::embedding::{Embedding, VectorSpace};
use floneumin_sample::Tokenizer;
use futures_util::{Stream, StreamExt};
use llm_samplers::prelude::Sampler;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;
use url::Url;

pub struct UnknownVectorSpace;

impl VectorSpace for UnknownVectorSpace {}

#[async_trait::async_trait]
pub trait Embedder<S: VectorSpace + Send + Sync + 'static>: Send + Sync + 'static {
    async fn embed(&self, input: &str) -> anyhow::Result<Embedding<S>>;

    async fn embed_batch(&self, inputs: &[&str]) -> anyhow::Result<Vec<Embedding<S>>>;

    fn into_any_embedder(self) -> DynEmbedder
    where
        Self: Sized,
    {
        Box::new(AnyEmbedder::<S, Self>(self, PhantomData))
    }
}

pub type DynEmbedder = Box<dyn Embedder<UnknownVectorSpace>>;

struct AnyEmbedder<S: VectorSpace + Send + Sync + 'static, E: Embedder<S> + Send + Sync + 'static>(
    E,
    PhantomData<S>,
);

#[async_trait::async_trait]
impl<S: VectorSpace + Send + Sync + 'static, E: Embedder<S> + Send + Sync + 'static>
    Embedder<UnknownVectorSpace> for AnyEmbedder<S, E>
{
    async fn embed(&self, input: &str) -> anyhow::Result<Embedding<UnknownVectorSpace>> {
        self.0.embed(input).await.map(|e| e.cast())
    }

    async fn embed_batch(
        &self,
        inputs: &[&str],
    ) -> anyhow::Result<Vec<Embedding<UnknownVectorSpace>>> {
        self.0
            .embed_batch(inputs)
            .await
            .map(|e| e.into_iter().map(|e| e.cast()).collect())
    }
}

#[async_trait::async_trait]
pub trait CreateModel {
    async fn start() -> Self;

    fn requires_download() -> bool;
}

#[async_trait::async_trait]
pub trait Model: 'static {
    type TextStream: Stream<Item = String> + Send + Sync + Unpin + 'static;

    fn tokenizer(&self) -> Arc<dyn Tokenizer + Send + Sync>;

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

    async fn generate_text(
        &mut self,
        prompt: &str,
        generation_parameters: GenerationParameters,
    ) -> anyhow::Result<String> {
        let mut text = String::new();

        let mut stream = self.stream_text(prompt, generation_parameters).await?;
        while let Some(new) = stream.next().await {
            text.push_str(&new);
        }
        Ok(text)
    }

    async fn stream_text_with_sampler(
        &mut self,
        _prompt: &str,
        _max_tokens: Option<u32>,
        _stop_on: Option<&str>,
        _sampler: Arc<Mutex<dyn Sampler<u32, f32>>>,
    ) -> anyhow::Result<Self::TextStream> {
        Err(anyhow::Error::msg("Not implemented"))
    }

    async fn stream_text(
        &mut self,
        prompt: &str,
        generation_parameters: crate::model::GenerationParameters,
    ) -> anyhow::Result<Self::TextStream>;

    fn into_any_model(self) -> DynModel
    where
        Self: Send + Sync + Sized,
    {
        Box::new(AnyModel(self, PhantomData))
    }
}

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

    async fn stream_text(
        &mut self,
        prompt: &str,
        generation_parameters: crate::model::GenerationParameters,
    ) -> anyhow::Result<Self::TextStream> {
        self.0
            .stream_text(prompt, generation_parameters)
            .await
            .map(|s| Box::new(s) as Box<dyn Stream<Item = String> + Send + Sync + Unpin>)
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

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = repetition_penalty;
        self
    }

    pub fn with_repetition_penalty_range(mut self, repetition_penalty_range: u32) -> Self {
        self.repetition_penalty_range = repetition_penalty_range;
        self
    }

    pub fn with_max_length(mut self, max_length: u32) -> Self {
        self.max_length = max_length;
        self
    }

    pub fn with_stop_on(mut self, stop_on: impl Into<Option<String>>) -> Self {
        self.stop_on = stop_on.into();
        self
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    pub fn top_k(&self) -> u32 {
        self.top_k
    }

    pub fn top_p(&self) -> f32 {
        self.top_p
    }

    pub fn repetition_penalty(&self) -> f32 {
        self.repetition_penalty
    }

    pub fn repetition_penalty_range(&self) -> u32 {
        self.repetition_penalty_range
    }

    pub fn max_length(&self) -> u32 {
        self.max_length
    }

    pub fn stop_on(&self) -> Option<&str> {
        self.stop_on.as_deref()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ModelType {
    Mpt(MptType),
    GptNeoX(GptNeoXType),
    Llama(LlamaType),
}

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

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum MptType {
    Base,
    Story,
    Instruct,
    Chat,
    Custom(Url),
}

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

struct CustomSpace<const URL: u128>;

impl<const URL: u128> VectorSpace for CustomSpace<URL> {}
