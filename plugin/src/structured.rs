use llm::{Sampler, TokenBias, TokenId, Tokenizer};
use partial_sort::PartialSort;
use rand::{distributions::WeightedIndex, prelude::Distribution};
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

impl<V: for<'a> Validate<'a> + Send + Sync> Sampler for StructuredSampler<V> {
    fn sample(
        &self,
        previous_tokens: &[TokenId],
        logits: &[f32],
        rng: &mut dyn rand::RngCore,
    ) -> TokenId {
        let Self {
            top_k,
            top_p,
            repeat_penalty,
            temperature,
            repetition_penalty_last_n,
            ..
        } = *self;
        let bias_tokens = &self.bias_tokens;

        let n_logits = logits.len();
        let mut logits_id = Vec::<(f32, TokenId)>::with_capacity(n_logits);

        // TODO: consider if this can be modularized and this sampler can be composed out of multiple pieces,
        // instead of having this monolithic function that embeds the repetition penalty and token bias
        {
            let scale = 1.0 / temperature;
            for (i, &logit) in logits.iter().enumerate() {
                let tid = i as TokenId;

                let val = if self.invalid_token(previous_tokens, tid) {
                    continue;
                } else if let Some(logit_override) = bias_tokens.get(tid) {
                    logit_override
                } else if previous_tokens[previous_tokens
                    .len()
                    .saturating_sub(repetition_penalty_last_n)..]
                    .contains(&(i as TokenId))
                {
                    // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
                    // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main

                    // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if logits[i] < 0.0 {
                        logit * scale * repeat_penalty
                    } else {
                        logit * scale / repeat_penalty
                    }
                } else {
                    logit * scale
                };
                logits_id.push((val, tid));
            }
        }

        // find the top K tokens
        {
            let logits_id_len = logits_id.len();
            logits_id.partial_sort(top_k.min(logits_id_len), |a, b| {
                // Sort descending
                b.0.total_cmp(&a.0)
            });
            logits_id.truncate(top_k);
        }

        let maxl = logits_id
            .iter()
            .map(|x| x.0)
            .max_by(f32::total_cmp)
            .unwrap();

        // compute probs for the top K tokens
        let mut probs: Vec<f32> = logits_id
            .iter()
            .copied()
            .map(|(k, _)| (k - maxl).exp())
            .collect();
        let sum: f32 = probs.iter().copied().sum();

        // Normalize the probs
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Top p sampling
        if top_p < 1.0 {
            let mut cumsum = 0.0;
            for i in 0..probs.len() {
                cumsum += probs[i];
                if cumsum >= top_p {
                    probs.truncate(i + 1);
                    logits_id.truncate(i + 1);
                    break;
                }
            }

            cumsum = 1.0 / cumsum;
            for p in probs.iter_mut() {
                *p *= cumsum;
            }
        }

        let dist = WeightedIndex::new(&probs).expect("WeightedIndex error");
        let idx = dist.sample(rng);

        logits_id[idx].1
    }
}
