use super::structured_parser::{ParseStatus, ParseStream, Validate};
use crate::sample::DynTokenizer;
use crate::sample::Tokenizer;
use llm_samplers::prelude::Sampler;
use llm_samplers::types::{HasSamplerResources, Logits};
use rustc_hash::FxHashMap;
use std::fmt::Debug;

pub struct StructuredSampler<V: for<'a> Validate<'a>> {
    pub(crate) structure: V,
    pub(crate) current_token_count: usize,
    pub(crate) tokenizer: DynTokenizer,
    cache: FxHashMap<Vec<u32>, String>,
}

impl<V: for<'a> Validate<'a>> StructuredSampler<V> {
    pub fn new(
        structure: V,
        current_token_count: usize,
        tokenizer: impl Into<DynTokenizer>,
    ) -> Self {
        let tokenizer = tokenizer.into();
        Self {
            structure,
            current_token_count,
            tokenizer,
            cache: FxHashMap::default(),
        }
    }

    fn invalid_token(&mut self, previous_tokens: &[u32], new_token: u32) -> bool {
        let tokens = &previous_tokens[self.current_token_count.saturating_sub(1)..];
        let tokens = self
            .cache
            .entry(tokens.to_vec())
            .or_insert_with(|| match self.tokenizer.decode(tokens) {
                Ok(tokens) => tokens,
                Err(_) => String::new(),
            })
            .as_str();

        let mut borrowed = vec![tokens];

        let new_token = match self.tokenizer.decode(&[new_token]) {
            Ok(tokens) => tokens,
            Err(_) => return true,
        };

        borrowed.push(new_token.as_str());

        if borrowed.iter().all(|s| s.is_empty()) {
            return true;
        }

        let status = self.structure.validate(ParseStream::new(&borrowed));

        match status {
            ParseStatus::Complete(Some(_)) => true,
            ParseStatus::Complete(None) => false,
            ParseStatus::Incomplete { .. } => new_token.is_empty(),
            ParseStatus::Invalid => true,
        }
    }
}

impl<V: for<'a> Validate<'a>> Debug for StructuredSampler<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StructuredSampler").finish()
    }
}

impl<V: for<'a> Validate<'a> + Send + Sync> Sampler<u32, f32> for StructuredSampler<V> {
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = u32>,
        logits: &'a mut Logits<u32, f32>,
    ) -> anyhow::Result<&'a mut Logits<u32, f32>> {
        res.with_last_tokens(&mut |previous_tokens| {
            logits.retain(|tid| !self.invalid_token(previous_tokens, tid.token_id))
        })?;
        Ok(logits)
    }
}
