use std::sync::Arc;
use tokenizers::Decoder;
use tokenizers::DecoderWrapper;
use tokenizers::Model;
use tokenizers::ModelWrapper;
use tokenizers::Normalizer;
use tokenizers::NormalizerWrapper;
use tokenizers::PostProcessor;
use tokenizers::PostProcessorWrapper;
use tokenizers::PreTokenizer;
use tokenizers::PreTokenizerWrapper;
use tokenizers::TokenizerImpl;

pub mod structured;
pub mod structured_parser;

pub struct DynTokenizer {
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
}

impl<M, N, PT, PP, D> From<tokenizers::tokenizer::TokenizerImpl<M, N, PT, PP, D>> for DynTokenizer
where
    M: Model + Send + Sync + 'static,
    N: Normalizer + Send + Sync + 'static,
    PT: PreTokenizer + Send + Sync + 'static,
    PP: PostProcessor + Send + Sync + 'static,
    D: Decoder + Send + Sync + 'static,
{
    fn from(tokenizer: tokenizers::tokenizer::TokenizerImpl<M, N, PT, PP, D>) -> Self {
        Self::new(tokenizer)
    }
}

impl From<tokenizers::Tokenizer> for DynTokenizer {
    fn from(tokenizer: tokenizers::Tokenizer) -> Self {
        Self::new(tokenizer)
    }
}

impl From<Arc<dyn Tokenizer + Send + Sync>> for DynTokenizer {
    fn from(tokenizer: Arc<dyn Tokenizer + Send + Sync>) -> Self {
        Self {
            tokenizer: tokenizer.clone(),
        }
    }
}

impl From<&llm::Tokenizer> for DynTokenizer {
    fn from(tokenizer: &llm::Tokenizer) -> Self {
        Self::new(match tokenizer {
            llm::Tokenizer::Embedded(embedded) => llm::Tokenizer::Embedded(embedded.clone()),
            llm::Tokenizer::HuggingFace(hugging_face) => {
                llm::Tokenizer::HuggingFace(hugging_face.clone())
            }
        })
    }
}

impl DynTokenizer {
    pub fn new<T: Tokenizer + Send + Sync + 'static>(tokenizer: T) -> Self {
        Self {
            tokenizer: Arc::new(tokenizer),
        }
    }
}

impl Tokenizer for DynTokenizer {
    fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.tokenizer.decode(ids)
    }
}

pub trait Tokenizer {
    fn decode(&self, ids: &[u32]) -> anyhow::Result<String>;
}

impl Tokenizer for llm::Tokenizer {
    fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        let bytes = self.decode(ids.into(), false);
        Ok(String::from_utf8(bytes).unwrap())
    }
}

impl<M, N, PT, PP, D> Tokenizer for tokenizers::tokenizer::TokenizerImpl<M, N, PT, PP, D>
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
    PP: PostProcessor,
    D: Decoder,
{
    fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        Ok(self
            .decode(ids.into(), false)
            .map_err(|e| anyhow::anyhow!(e))?)
    }
}

impl Tokenizer for tokenizers::Tokenizer {
    fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        let as_impl: &TokenizerImpl<
            ModelWrapper,
            NormalizerWrapper,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
        > = &*self;
        Ok(as_impl
            .decode(ids.into(), false)
            .map_err(|e| anyhow::anyhow!(e))?)
    }
}
