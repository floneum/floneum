use std::{
    fmt::{Debug, Display, Formatter},
    sync::{Arc, Mutex},
};

use crate::SyncModel;
use crate::TokenOutputStream;
use kalosm_sample::{ParseStatus, Parser, Tokenizer};
use llm_samplers::prelude::{Logit, Logits};
use llm_samplers::types::{HasSamplerResources, Sampler, SamplerError};
use rustc_hash::FxHashMap;

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
    let prompt_tokens = tokenizer.encode(&prompt_text, true)?;
    let mut unprocessed_token_count = prompt_tokens.len();
    let mut token_stream = TokenOutputStream::new(tokenizer.clone());
    for token in prompt_tokens {
        token_stream.next_token(token)?;
    }
    let mut rng = rand::thread_rng();

    loop {
        let tokens = token_stream.tokens();
        let logit_probs =
            llm.feed_tokens(session, &tokens[tokens.len() - unprocessed_token_count..])?;
        let resources = &mut SamplerResources {
            previous_tokens: tokens,
            rng: &mut rng,
        };
        let mut state_map = FxHashMap::default();

        let mut logits = Logits::default();

        let next_tokens = token_stream.peek_tokens((0..logit_probs.len() as u32).collect())?;

        let mut logits_indexed = (0..)
            .zip(logit_probs.iter().copied())
            .map(|(id, prob)| Logit {
                token_id: id as u32,
                logit: prob,
                prob: 0f32,
            })
            .collect::<Vec<_>>();

        // We can partition the logits into two groups: the top k and the rest
        if let Some(top_k) = top_k {
            if top_k < logits_indexed.len() {
                logits_indexed
                    .select_nth_unstable_by(top_k, |a, b| b.logit.partial_cmp(&a.logit).unwrap());
            }
        }

        for i in 0..logits_indexed.len() {
            let Logit {
                token_id, logit, ..
            } = logits_indexed[i];
            let Some(text) = next_tokens[token_id as usize].as_ref() else {
                continue;
            };
            if let Ok(result) = parser.parse(&parser_state, text.as_bytes()) {
                let parsed_bytes = match result {
                    ParseStatus::Finished { remaining, .. } => text.len() - remaining.len(),
                    ParseStatus::Incomplete { .. } => text.len(),
                };
                let result = result.without_remaining();
                state_map.insert(token_id, (result, parsed_bytes));
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
            } else if let Some(top_k) = top_k {
                // If the remaining logits are less than the top k, no need to partition
                let remaining_needed = (top_k - logits.len()).min(logits_indexed.len() - i);
                let remaining_possible = logits_indexed.len() - i;
                if remaining_possible < remaining_needed {
                    // If we eliminated a logit, our partitioning of the logits is no longer valid
                    logits_indexed[i..].select_nth_unstable_by(remaining_needed, |a, b| {
                        b.logit.partial_cmp(&a.logit).unwrap()
                    });
                }
            }
        }

        // If there are no valid tokens, return an error
        if state_map.is_empty() {
            return Err(anyhow::anyhow!("No valid tokens found"));
        }
        let token_id = sampler
            .sample_token(resources, &mut logits)?
            .ok_or(anyhow::anyhow!("Failed to sample constrained tokens"))?;

        unprocessed_token_count = 1;
        let (result, parsed_bytes) = state_map
            .remove(&token_id)
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
    tokenizer: &Arc<dyn Tokenizer + Send + Sync>,
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
                tracing::trace!("Required next: {}", required_next);
                let mut extra_tokens = tokenizer.encode(&required_next, false)?;
                // Remove the last token to avoid influencing the next token
                extra_tokens.pop();
                if extra_tokens.is_empty() {
                    return Ok(None);
                }
                let required_next = tokenizer.decode(&extra_tokens)?;
                if required_next.is_empty() {
                    return Ok(None);
                }
                let result = parser
                    .parse(parser_state, required_next.as_bytes())
                    .unwrap_or_else(|_| {
                        unreachable!("Required next should always be valid attempted to add {} but got error", required_next)
                });
                for token in extra_tokens {
                    if let Some(token) = token_stream.next_token(token)? {
                        on_token(token)?;
                    }

                    *unprocessed_token_count += 1;
                }
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
