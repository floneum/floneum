use rustc_hash::FxHashMap;
use std::borrow::Cow;
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
    fn decode(&self, ids: &[u32]) -> anyhow::Result<Cow<'_, str>> {
        self.tokenizer.decode(ids)
    }
}

pub trait Tokenizer {
    fn decode(&self, ids: &[u32]) -> anyhow::Result<Cow<'_, str>>;

    fn decode_batch(&self, ids: &[&[u32]]) -> anyhow::Result<Vec<Cow<'_, str>>> {
        ids.iter().map(|id| self.decode(id)).collect()
    }
}

impl Tokenizer for llm::Tokenizer {
    fn decode(&self, ids: &[u32]) -> anyhow::Result<Cow<'_, str>> {
        let bytes = self.decode(ids.into(), false);
        Ok(String::from_utf8(bytes)?.into())
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
    fn decode(&self, ids: &[u32]) -> anyhow::Result<Cow<'_, str>> {
        Ok(self
            .decode(ids.into(), false)
            .map(|s| s.into())
            .map_err(|e| anyhow::anyhow!(e))?)
    }
}

pub struct FasterHuggingFaceTokenizer {
    inner: tokenizers::Tokenizer,
    single_token_map: FxHashMap<u32, Cow<'static, str>>,
}

impl FasterHuggingFaceTokenizer {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        let single_token_map = tokenizer
            .get_vocab(true)
            .into_iter()
            .map(|(string, token_id)| {
                let decoded = if let Some(decoder) = tokenizer.get_decoder() {
                    decoder.decode(vec![string]).unwrap()
                } else {
                    string
                };
                (token_id, decoded.into())
            })
            .collect();
        Self {
            inner: tokenizer,
            single_token_map,
        }
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }

    pub fn tokenizer_mut(&mut self) -> &mut tokenizers::Tokenizer {
        &mut self.inner
    }

    pub fn into_tokenizer(self) -> tokenizers::Tokenizer {
        self.inner
    }
}

impl Tokenizer for FasterHuggingFaceTokenizer {
    fn decode(&self, ids: &[u32]) -> anyhow::Result<Cow<'_, str>> {
        if ids.len() == 1 {
            if let Some(token) = self.single_token_map.get(&ids[0]) {
                return Ok(token.clone());
            }
        }
        let mut tokens = String::new();
        for id in ids {
            tokens.push_str(
                self.single_token_map
                    .get(id)
                    .clone().map(|s| &**s)
                    .unwrap_or_else(|| "".into())
            );
        }
        Ok(tokens.into())
    }

    fn decode_batch(&self, ids: &[&[u32]]) -> anyhow::Result<Vec<Cow<'_, str>>> {
        let mut tokens = Vec::with_capacity(ids.len());
        for id in ids {
            if id.len() == 1 {
                if let Some(token) = self.single_token_map.get(&id[0]) {
                    tokens.push(token.clone());
                    continue;
                }
            }
            let mut token = String::new();
            for id in *id {
                token.push_str(
                    self
                        .single_token_map
                        .get(id)
                        .clone().map(|s| &**s)
                        .unwrap_or( "")
                );
            }
            tokens.push(token.into());
        }
        Ok(tokens)
    }
}

impl Tokenizer for tokenizers::Tokenizer {
    fn decode(&self, ids: &[u32]) -> anyhow::Result<Cow<'_, str>> {
        let as_impl: &TokenizerImpl<
            ModelWrapper,
            NormalizerWrapper,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
        > = &*self;
        Ok(as_impl
            .decode(ids.into(), false)
            .map_err(|e| anyhow::anyhow!(e))?
            .into())
    }
}
