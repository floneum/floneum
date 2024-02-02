//! # kalosm-sample
//! This is a sampling library for Kalosm.
//!
//! It handles choosing a token from a probability distribution. Samplers can be used to constrain the generation of text for example you can use a sampler to prevent the model from generating the same word twice in a row. Or you could only allow the model to generate a list of single digit numbers.

#![warn(missing_docs)]

use std::borrow::Cow;
use std::ops::Deref;
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

mod structured_parser;
pub use structured_parser::*;
#[cfg(feature = "llamacpp")]
mod llm;

/// A type erased wrapper for a tokenizer.
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

impl DynTokenizer {
    /// Create a new `DynTokenizer` from a `Tokenizer`.
    pub fn new<T: Tokenizer + Send + Sync + 'static>(tokenizer: T) -> Self {
        Self {
            tokenizer: Arc::new(tokenizer),
        }
    }
}

impl Tokenizer for DynTokenizer {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        self.tokenizer.encode(text)
    }

    fn decode(&self, ids: &[u32]) -> anyhow::Result<Cow<'_, str>> {
        self.tokenizer.decode(ids)
    }

    fn get_all_tokens(&self) -> anyhow::Result<Cow<'_, [u32]>> {
        self.tokenizer.get_all_tokens()
    }
}

/// A tokenizer is a type that can decode a list of token ids into a string.
pub trait Tokenizer {
    /// Encode a string into a list of token ids.
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>>;

    /// Encode a list of strings into a list of token ids.
    fn encode_batch(&self, text: &[&str]) -> anyhow::Result<Vec<Vec<u32>>> {
        text.iter().map(|text| self.encode(text)).collect()
    }

    /// Decode a list of token ids into a string.
    fn decode(&self, ids: &[u32]) -> anyhow::Result<Cow<'_, str>>;

    /// Decode a list of a list of token ids into a string.
    fn decode_batch(&self, ids: &[&[u32]]) -> anyhow::Result<Vec<Cow<'_, str>>> {
        ids.iter().map(|id| self.decode(id)).collect()
    }

    /// Get all possible tokens.
    fn get_all_tokens(&self) -> anyhow::Result<Cow<'_, [u32]>>;
}

impl<M, N, PT, PP, D> Tokenizer for tokenizers::tokenizer::TokenizerImpl<M, N, PT, PP, D>
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
    PP: PostProcessor,
    D: Decoder,
{
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        Ok(self
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!(e))?
            .get_ids()
            .to_vec())
    }

    fn decode(&self, ids: &[u32]) -> anyhow::Result<Cow<'_, str>> {
        self.decode(ids, false)
            .map(|s| s.into())
            .map_err(|e| anyhow::anyhow!(e))
    }

    fn get_all_tokens(&self) -> anyhow::Result<Cow<'_, [u32]>> {
        Ok(self
            .get_vocab(true)
            .into_values()
            .collect::<Vec<_>>()
            .into())
    }
}

/// A tokenizer that uses the HuggingFace tokenizer with a cache for single tokens.
pub struct FasterHuggingFaceTokenizer {
    inner: tokenizers::Tokenizer,
    all_tokens: Vec<u32>,
}

impl FasterHuggingFaceTokenizer {
    /// Create a new `FasterHuggingFaceTokenizer` from a `tokenizers::Tokenizer`.
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            all_tokens: tokenizer.get_vocab(true).into_values().collect(),
            inner: tokenizer,
        }
    }

    /// Get the inner tokenizer.
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }

    /// Get the inner tokenizer mutably.
    pub fn tokenizer_mut(&mut self) -> &mut tokenizers::Tokenizer {
        &mut self.inner
    }

    /// Consume the `FasterHuggingFaceTokenizer` and return the inner tokenizer.
    pub fn into_tokenizer(self) -> tokenizers::Tokenizer {
        self.inner
    }
}

impl Tokenizer for FasterHuggingFaceTokenizer {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        self.inner.encode(text)
    }

    fn decode(&self, ids: &[u32]) -> anyhow::Result<Cow<'_, str>> {
        self.inner.decode(ids)
    }

    fn decode_batch(&self, ids: &[&[u32]]) -> anyhow::Result<Vec<Cow<'_, str>>> {
        self.inner.decode_batch(ids)
    }

    fn get_all_tokens(&self) -> anyhow::Result<Cow<'_, [u32]>> {
        Ok((&self.all_tokens).into())
    }
}

impl Tokenizer for tokenizers::Tokenizer {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let deref = self.deref();
        Ok(deref
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!(e))?
            .get_ids()
            .to_vec())
    }

    fn decode(&self, ids: &[u32]) -> anyhow::Result<Cow<'_, str>> {
        let as_impl: &TokenizerImpl<
            ModelWrapper,
            NormalizerWrapper,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
        > = self;
        Ok(as_impl
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!(e))?
            .into())
    }

    fn get_all_tokens(&self) -> anyhow::Result<Cow<'_, [u32]>> {
        let as_impl: &TokenizerImpl<
            ModelWrapper,
            NormalizerWrapper,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
        > = self;
        Ok(as_impl
            .get_vocab(true)
            .into_values()
            .collect::<Vec<_>>()
            .into())
    }
}
