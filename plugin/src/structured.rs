use llm::{TokenBias, TokenId, Tokenizer};
use llm_samplers::types::{HasSamplerResources, Logits};
use std::fmt::Debug;

use crate::structured_parser::{ParseStatus, ParseStream, Validate};

pub struct StructuredSampler<V: for<'a> Validate<'a>> {
    pub tokenizer: Tokenizer,
    pub structure: V,
    /// The top K words by score are kept during sampling.
    top_k: usize,
    /// The cumulative probability after which no more words are kept for sampling.
    top_p: f32,
    /// The penalty for repeating tokens. Higher values make the generation less
    /// likely to get into a loop, but may harm results when repetitive outputs
    /// are desired.
    repeat_penalty: f32,
    /// Temperature (randomness) used for sampling. A higher number is more random.
    temperature: f32,
    /// A list of tokens to bias against in the process of generation.
    bias_tokens: TokenBias,
    /// The number of tokens to consider for the repetition penalty.
    repetition_penalty_last_n: usize,
    pub current_token_count: usize,
}

impl<V: for<'a> Validate<'a>> StructuredSampler<V> {
    pub fn new(tokenizer: Tokenizer, structure: V, current_token_count: usize) -> Self {
        Self {
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.30,
            temperature: 0.80,
            bias_tokens: TokenBias::empty(),
            repetition_penalty_last_n: 512,
            tokenizer,
            structure,
            current_token_count,
        }
    }

    fn invalid_token(&self, previous_tokens: &[TokenId], new_token: TokenId) -> bool {
        let mut tokens = Vec::new();
        for token in &previous_tokens[self.current_token_count.saturating_sub(1)..] {
            let token = self.tokenizer.token(*token as usize);
            let Ok(token) = String::from_utf8(token) else {
                return true;
            };
            if !token.is_ascii() {
                return true;
            }
            tokens.push(token);
        }

        let mut borrowed = tokens.iter().map(|x| x.as_str()).collect::<Vec<_>>();

        let new_token = self.tokenizer.token(new_token as usize);
        let Ok(new_token) = String::from_utf8(new_token) else {
            return true;
        };
        if !new_token.is_ascii() {
            return true;
        }

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
        f.debug_struct("StructuredSampler")
            .field("top_k", &self.top_k)
            .field("top_p", &self.top_p)
            .field("repeat_penalty", &self.repeat_penalty)
            .field("temperature", &self.temperature)
            .field("bias_tokens", &self.bias_tokens)
            .field("repetition_penalty_last_n", &self.repetition_penalty_last_n)
            .finish()
    }
}

impl<V: for<'a> Validate<'a> + Send + Sync> llm_samplers::prelude::Sampler<TokenId, f32>
    for StructuredSampler<V>
{
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TokenId>,
        logits: &'a mut Logits<TokenId, f32>,
    ) -> anyhow::Result<&'a mut Logits<TokenId, f32>> {
        res.with_last_tokens(&mut |previous_tokens| {
            logits.retain(|tid| !self.invalid_token(previous_tokens, tid.token_id))
        })?;
        Ok(logits)
    }
}
