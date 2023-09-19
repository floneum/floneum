use crate::{
    embedding::{Embedding, VectorSpace},
    structured_parser::Validate,
};
use futures_util::Stream;
use url::Url;

#[async_trait::async_trait]
pub trait Model<S: VectorSpace>: 'static {
    type TextStream: Stream<Item = String> + Send + Sync + Unpin + 'static;

    async fn start() -> Self;

    async fn embed(input: &str) -> anyhow::Result<Embedding<S>>;

    async fn embed_batch(inputs: &[&str]) -> anyhow::Result<Vec<Embedding<S>>>;

    async fn generate_text(
        &mut self,
        prompt: &str,
        generation_parameters: GenerationParameters,
    ) -> anyhow::Result<String>;

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
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub repetition_penalty_range: u32,
    pub repetition_penalty_slope: f32,
    pub max_length: u32,
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
