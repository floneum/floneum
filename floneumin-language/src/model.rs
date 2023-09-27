use crate::{
    embedding::{Embedding, VectorSpace},
    structured_parser::Validate,
};
use futures_util::{Stream, StreamExt};
use url::Url;

#[async_trait::async_trait]
pub trait Embedder<S: VectorSpace>: 'static {
    async fn embed(input: &str) -> anyhow::Result<Embedding<S>>;

    async fn embed_batch(inputs: &[&str]) -> anyhow::Result<Vec<Embedding<S>>>;
}

#[async_trait::async_trait]
pub trait Model: 'static {
    type TextStream: Stream<Item = String> + Send + Sync + Unpin + 'static;

    async fn start() -> Self;

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

    async fn stream_text(
        &mut self,
        prompt: &str,
        generation_parameters: crate::model::GenerationParameters,
    ) -> anyhow::Result<Self::TextStream>;

    async fn infer_validate<V: for<'a> Validate<'a> + Clone + Send + Sync + 'static>(
        &mut self,
        _prompt: String,
        _max_tokens: Option<u32>,
        _validator: V,
    ) -> String {
        panic!("{}", std::any::type_name::<Self>())
    }
}

pub struct GenerationParameters {
    temperature: f32,
    top_k: u32,
    top_p: f32,
    repetition_penalty: f32,
    repetition_penalty_range: u32,
    repetition_penalty_slope: f32,
    max_length: u32,
}

impl Default for GenerationParameters {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.9,
            repetition_penalty: 1.0,
            repetition_penalty_range: 0,
            repetition_penalty_slope: 0.0,
            max_length: 128,
        }
    }
}

impl GenerationParameters {
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

    pub fn with_repetition_penalty_slope(mut self, repetition_penalty_slope: f32) -> Self {
        self.repetition_penalty_slope = repetition_penalty_slope;
        self
    }

    pub fn with_max_length(mut self, max_length: u32) -> Self {
        self.max_length = max_length;
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

    pub fn repetition_penalty_slope(&self) -> f32 {
        self.repetition_penalty_slope
    }

    pub fn max_length(&self) -> u32 {
        self.max_length
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
embedding!(BaseSpace);
embedding!(StorySpace);
embedding!(InstructSpace);
embedding!(ChatSpace);
embedding!(LargePythiaSpace);
embedding!(TinyPythiaSpace);
embedding!(DollySevenBSpace);
embedding!(StableLmSpace);

struct CustomSpace<const URL: u128>;

impl<const URL: u128> VectorSpace for CustomSpace<URL> {}
