use std::{
    fmt::{Debug, Display, Formatter},
    sync::{Arc, Mutex},
};

use crate::SyncModel;
use crate::TokenOutputStream;
use kalosm_sample::{ParseStatus, Parser};
use llm_samplers::prelude::{Logit, Logits};
use llm_samplers::types::{HasSamplerResources, Sampler, SamplerError};
use tokenizers::tokenizer::Tokenizer;

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_structured<M: ?Sized + SyncModel, P: Parser>(
    prompt: impl Display,
    llm: &M,
    session: &mut M::Session,
    parser: P,
    mut parser_state: P::PartialState,
    mut sampler: Arc<Mutex<dyn Sampler>>,
    mut on_token: impl FnMut(String) -> anyhow::Result<()>,
    top_k: Option<usize>,
) -> anyhow::Result<P::Output> {
    let tokenizer = llm.tokenizer();

    let prompt_text = prompt.to_string();
    let prompt_tokens = tokenizer
        .encode(prompt_text, true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let prompt_tokens = prompt_tokens.get_ids();
    let mut unprocessed_token_count = prompt_tokens.len();
    let mut token_stream = TokenOutputStream::new(tokenizer.clone());
    for token in prompt_tokens {
        token_stream.next_token(*token)?;
    }
    let mut rng = rand::thread_rng();
    let mut state_map = vec![];

    loop {
        let tokens = token_stream.tokens();
        let logit_probs =
            llm.feed_tokens(session, &tokens[tokens.len() - unprocessed_token_count..])?;
        let resources = &mut SamplerResources {
            previous_tokens: tokens,
            rng: &mut rng,
        };

        let mut logits = Logits::default();

        let mut token_cache = DetokenizationCache::new(logit_probs.len());

        let mut logits_indexed = (0..)
            .zip(logit_probs.iter().copied())
            .map(|(id, prob)| Logit {
                token_id: id as u32,
                logit: prob,
                prob: 0f32,
            })
            .collect::<Vec<_>>();

        // fill the state map with None for each token
        state_map.clear();
        for _ in 0..logits_indexed.len() {
            state_map.push(None);
        }
        let mut valid_tokens = false;

        // If we don't have a top k, then we can just cache the entire detokenization
        if top_k.is_none() {
            token_cache.expand(
                &(0..logit_probs.len() as u32).collect::<Vec<_>>(),
                &token_stream,
            )?;
        }

        const DETOKENIZATION_INITIAL_BATCH_SIZE: usize = 64;

        // Constraints tend to be either very difficult to satisfy or very easy to satisfy
        // We exponentially increase the batch size as a balance between the two
        // If the first half of the tokens are invalid, it is unlikely that the first 64 tokens of the second half will be valid
        let mut detokenization_batch_size = DETOKENIZATION_INITIAL_BATCH_SIZE;

        let mut partitioned_logits_index = top_k.map(|_| 0);

        for i in 0..logits_indexed.len() {
            // If we have top k enabled, and there are less than top k - committed logits sorted, we need to expand the partitioned logits
            if let (Some(top_k), Some(partitioned_index)) = (top_k, partitioned_logits_index) {
                // If the remaining logits are less than the top k, no need to partition
                let remaining_needed = top_k - logits.len();
                let remaining_possible = partitioned_index - i;
                if remaining_possible <= remaining_needed {
                    // We batch together updates to the cache by DETOKENIZATION_BATCH_SIZE
                    let logits_to_update = (remaining_needed.max(detokenization_batch_size))
                        .min(logits_indexed.len() - 1 - i);
                    let new_partitioned_index = i + logits_to_update;

                    // If we eliminated a logit, our partitioning of the logits is no longer valid
                    logits_indexed[i..].select_nth_unstable_by(logits_to_update, |a, b| {
                        b.logit.partial_cmp(&a.logit).unwrap()
                    });
                    // Expand the cache to include the new logits
                    partitioned_logits_index = Some(new_partitioned_index);
                    let tokens = logits_indexed[i..=new_partitioned_index]
                        .iter()
                        .map(|logit| logit.token_id)
                        .collect::<Vec<_>>();
                    token_cache.expand(&tokens, &token_stream)?;

                    // Double the batch size for next time
                    detokenization_batch_size = detokenization_batch_size.saturating_mul(4);
                }
            }

            let Logit {
                token_id, logit, ..
            } = logits_indexed[i];
            let Some(text) = token_cache.get(token_id as usize) else {
                continue;
            };
            if let Ok(result) = parser.parse(&parser_state, text.as_bytes()) {
                let parsed_bytes = match result {
                    ParseStatus::Finished { remaining, .. } => text.len() - remaining.len(),
                    ParseStatus::Incomplete { .. } => text.len(),
                };
                let result = result.without_remaining();
                state_map[token_id as usize] = Some((result, parsed_bytes));
                valid_tokens = true;
                logits.push(Logit {
                    token_id,
                    logit,
                    prob: 0f32,
                });
                // If we only need to keep the top k logits, then we can quit early once we have enough
                if let Some(top_k) = top_k {
                    if logits.len() >= top_k {
                        break;
                    }
                }
            }
        }

        // If there are no valid tokens, return an error
        if !valid_tokens {
            return Err(anyhow::anyhow!("No valid tokens found"));
        }
        let token_id = sampler
            .sample_token(resources, &mut logits)?
            .ok_or(anyhow::anyhow!("Failed to sample constrained tokens"))?;

        unprocessed_token_count = 1;
        let (result, parsed_bytes) = state_map
            .get_mut(token_id as usize)
            .unwrap()
            .take()
            .ok_or(anyhow::anyhow!("Token {} not found in state map", token_id))?;
        let mut token = token_stream.next_token(token_id)?.unwrap();
        token.truncate(parsed_bytes);
        tracing::trace!("Adding token {} to parser", token);
        on_token(token)?;

        if let Some(result) = update_state(
            &parser,
            &mut parser_state,
            result,
            &tokenizer,
            &mut token_stream,
            &mut on_token,
            &mut unprocessed_token_count,
        )? {
            return Ok(result);
        }
    }
}

#[allow(unused, clippy::all)]
fn update_state<P: Parser>(
    parser: &P,
    parser_state: &mut P::PartialState,
    result: ParseStatus<P::PartialState, P::Output>,
    tokenizer: &Tokenizer,
    token_stream: &mut TokenOutputStream,
    on_token: &mut impl FnMut(String) -> anyhow::Result<()>,
    unprocessed_token_count: &mut usize,
) -> anyhow::Result<Option<P::Output>> {
    match result {
        kalosm_sample::ParseStatus::Incomplete {
            new_state,
            required_next,
        } => {
            *parser_state = new_state;
            if required_next.is_empty() {
                Ok(None)
            } else {
                // The token may decode to a string that is a valid prefix of the required next token, but in a way that doesn't let us decode the required next tokens
                let Some(mut extra_tokens) = token_stream.encode_after(&required_next)? else {
                    return Ok(None);
                };
                // Remove the last token to avoid influencing the next token
                extra_tokens.pop();
                // If there are no new tokens, continue generating tokens normally
                if extra_tokens.is_empty() {
                    return Ok(None);
                }

                let mut all_required_next = String::new();
                for token in extra_tokens {
                    if let Some(token) = token_stream.next_token(token)? {
                        all_required_next += &token;
                        on_token(token)?;
                    }

                    *unprocessed_token_count += 1;
                }
                let mut result = parser
                    .parse(parser_state, all_required_next.as_bytes())
                    .unwrap_or_else(|_| {
                        unreachable!("Required next should always be valid attempted to add {} but got error", required_next)
                });
                update_state(
                    parser,
                    parser_state,
                    result,
                    tokenizer,
                    token_stream,
                    on_token,
                    unprocessed_token_count,
                )
            }
        }
        kalosm_sample::ParseStatus::Finished { result, .. } => Ok(Some(result)),
    }
}

struct SamplerResources<'a, 'b, R: rand::Rng> {
    rng: &'a mut R,
    previous_tokens: &'b [u32],
}

impl<R> Debug for SamplerResources<'_, '_, R>
where
    R: rand::Rng,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplerResources")
            .field("previous_tokens", &self.previous_tokens)
            .finish()
    }
}

impl<R> HasSamplerResources for SamplerResources<'_, '_, R>
where
    R: rand::Rng,
{
    fn with_rng_mut(
        &mut self,
        fun: &mut dyn FnMut(&mut dyn rand::RngCore),
    ) -> Result<(), SamplerError> {
        fun(self.rng);
        Ok(())
    }

    fn with_last_tokens(&self, fun: &mut dyn FnMut(&[u32])) -> Result<(), SamplerError> {
        fun(self.previous_tokens);
        Ok(())
    }
}

#[derive(Clone)]
enum TokenCacheStatus {
    Empty,
    Invalid,
    Valid(String),
}

struct DetokenizationCache {
    cache: Box<[TokenCacheStatus]>,
}

impl DetokenizationCache {
    fn new(len: usize) -> Self {
        Self {
            cache: vec![TokenCacheStatus::Empty; len].into_boxed_slice(),
        }
    }

    fn get(&self, index: usize) -> Option<&str> {
        match &self.cache[index] {
            TokenCacheStatus::Empty => panic!("cache for token {} is empty", index),
            TokenCacheStatus::Invalid => None,
            TokenCacheStatus::Valid(token) => Some(token),
        }
    }

    fn expand(&mut self, tokens: &[u32], stream: &TokenOutputStream) -> anyhow::Result<()> {
        let new_tokens = stream.peek_tokens(tokens)?;

        for (&i, token) in tokens.iter().zip(new_tokens.into_iter()) {
            self.cache[i as usize] = match token {
                Some(token) => TokenCacheStatus::Valid(token),
                None => TokenCacheStatus::Invalid,
            };
        }

        Ok(())
    }
}
