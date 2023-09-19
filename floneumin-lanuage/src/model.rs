use crate::{
    embedding::{Embedding, VectorSpace},
    structured_parser::Validate,
};
use url::Url;

/// A type-erased model to allow for interacting with a model without knowing
/// its hyperparameters.
pub trait Model<S: VectorSpace>: 'static {
    fn start() -> Self;

    fn embed(input: &str) -> anyhow::Result<Embedding<S>>;

    fn embed_batch(inputs: &[&str]) -> anyhow::Result<Vec<Embedding<S>>> {
        inputs.iter().map(|input| Self::embed(input)).collect()
    }

    fn generate_text(
        &mut self,
        prompt: &str,
        generation_parameters: GenerationParameters,
    ) -> anyhow::Result<String>;

    fn infer_validate<V: for<'a> Validate<'a> + Clone + Send + Sync + 'static>(
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

#[derive(Debug, Clone)]
pub enum ModelType {
    Mpt(MptType),
    GptNeoX(GptNeoXType),
    Llama(LlamaType),
}

#[derive(Debug, Clone)]
pub enum LlamaType {
    Vicuna,
    Guanaco,
    WizardLm,
    Orca,
    LlamaSevenChat,
    LlamaThirteenChat,
    Custom(Url),
}

#[derive(Debug, Clone)]
pub enum MptType {
    Base,
    Story,
    Instruct,
    Chat,
    Custom(Url),
}

#[derive(Debug, Clone)]
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
