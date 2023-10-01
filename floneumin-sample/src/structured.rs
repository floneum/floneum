use super::structured_parser::{ParseStatus, ParseStream, Validate};
use crate::DynTokenizer;
use crate::Tokenizer;
use llm_samplers::prelude::Sampler;
use llm_samplers::types::{HasSamplerResources, Logits};
use std::fmt::Debug;

pub struct StructuredSampler<V: for<'a> Validate<'a>> {
    pub(crate) structure: V,
    pub(crate) current_token_count: usize,
    pub(crate) tokenizer: DynTokenizer,
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
            let tokens = &previous_tokens[self.current_token_count.saturating_sub(1)..];
            let tokens = match self.tokenizer.decode(tokens) {
                Ok(tokens) => tokens,
                Err(_) => String::new().into(),
            };

            let single_tokens = logits.iter().map(|tid| [tid.token_id]).collect::<Vec<_>>();
            let single_tokens_ref = single_tokens
                .iter()
                .map(|v| v.as_slice())
                .collect::<Vec<_>>();
            let new_tokens = self.tokenizer.decode_batch(&*single_tokens_ref).unwrap();
            let mut new_tokens = new_tokens.into_iter();

            logits.retain(|_| {
                let new_token = new_tokens.next().unwrap();
                let string = tokens.to_string() + &new_token;

                if string.is_empty() {
                    return true;
                }

                let status = self.structure.validate(ParseStream::new(&string));

                match status {
                    ParseStatus::Complete(Some(_)) => false,
                    ParseStatus::Complete(None) => true,
                    ParseStatus::Incomplete { .. } => !new_token.is_empty(),
                    ParseStatus::Invalid => false,
                }
            });
        })?;
        Ok(logits)
    }
}
