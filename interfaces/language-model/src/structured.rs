use std::{
    fmt::{Debug, Display, Formatter},
    sync::{Arc, Mutex},
};

use kalosm_sample::{ParseResult, Parser, Tokenizer};
use llm_samplers::types::{HasSamplerResources, Sampler, SamplerError};
use rustc_hash::FxHashMap;

use crate::SyncModel;

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_structured<M: ?Sized + SyncModel, P: Parser>(
    prompt: impl Display,
    llm: &mut M,
    session: &mut M::Session,
    tokenizer: &Arc<dyn Tokenizer + Send + Sync>,
    parser: P,
    mut parser_state: P::PartialState,
    mut sampler: Arc<Mutex<dyn Sampler>>,
    mut on_token: impl FnMut(String) -> anyhow::Result<()>,
) -> anyhow::Result<P::Output> {
    let prompt_text = prompt.to_string();
    let mut tokens = tokenizer.encode(&prompt_text)?;
    let mut prev_index = tokens.len();
    let mut current_index = tokens.len();
    let mut unprocessed_token_count = tokens.len();
    let mut rng = rand::thread_rng();

    loop {
        let mut logits =
            llm.feed_tokens(session, &tokens[tokens.len() - unprocessed_token_count..])?;
        let resources = &mut SamplerResources {
            previous_tokens: &tokens,
            rng: &mut rng,
        };
        let mut state_map = FxHashMap::default();
        let prev_text = if tokens.is_empty() {
            "".into()
        } else {
            let tokens = &tokens[prev_index..current_index];
            tokenizer.decode(tokens)?
        };
        for logit in logits.iter_mut() {
            let mut potential_new_tokens = tokens[prev_index..].to_vec();
            potential_new_tokens.push(logit.token_id);
            let token_text = tokenizer.decode(&potential_new_tokens)?;
            if token_text.len() > prev_text.len() {
                let text = token_text.split_at(prev_text.len());
                let new_text = text.1.to_string();
                if new_text.is_empty() {
                    continue;
                }
                if let Ok(result) = parser.parse(&parser_state, new_text.as_bytes()) {
                    let result = result.without_remaining();
                    state_map.insert(logit.token_id, Some((new_text.to_string(), result)));
                } else {
                    logit.logit = f32::NEG_INFINITY;
                }
            } else {
                state_map.insert(logit.token_id, None);
            }
        }
        if state_map.is_empty() {
            return Err(anyhow::anyhow!("No valid tokens found"));
        }
        let token_id = sampler
            .sample_token(resources, &mut logits)?
            .ok_or(anyhow::anyhow!("Failed to sample constrained tokens"))?;
        unprocessed_token_count = 1;
        tokens.push(token_id);
        if let Some((token, result)) = state_map
            .remove(&token_id)
            .ok_or(anyhow::anyhow!("Token {} not found in state map", token_id))?
        {
            tracing::trace!("Adding token {} to parser", token);
            prev_index = current_index;
            current_index = tokens.len();
            on_token(token.clone())?;

            if let Some(result) = update_state(
                &parser,
                &mut parser_state,
                result,
                tokenizer,
                &mut tokens,
                &mut on_token,
                &mut unprocessed_token_count,
            )? {
                return Ok(result);
            }
        }
    }
}

#[allow(unused, clippy::all)]
fn update_state<P: Parser>(
    parser: &P,
    parser_state: &mut P::PartialState,
    result: ParseResult<P::PartialState, P::Output>,
    tokenizer: &Arc<dyn Tokenizer + Send + Sync>,
    tokens: &mut Vec<u32>,
    on_token: &mut impl FnMut(String) -> anyhow::Result<()>,
    unprocessed_token_count: &mut usize,
) -> anyhow::Result<Option<P::Output>> {
    match result {
        kalosm_sample::ParseResult::Incomplete {
            new_state,
            required_next,
        } => {
            *parser_state = new_state;
            // TODO: restore faster required_next tokenization
            // if required_next.is_empty() {
            //     Ok(None)
            // } else {
            //     tracing::trace!("Required next: {}", required_next);
            //     let result = parser
            //         .parse(parser_state, required_next.as_bytes())
            //         .unwrap_or_else(|_| {
            //             unreachable!("Required next should always be valid attempted to add {} but got error", required_next)
            // });
            //     let extra_tokens = tokenizer.encode(&required_next)?;
            //     *unprocessed_token_count += extra_tokens.len();
            //     tokens.extend(extra_tokens);
            //     on_token(required_next.to_string())?;
            //     update_state(
            //         parser,
            //         parser_state,
            //         result,
            //         tokenizer,
            //         tokens,
            //         on_token,
            //         unprocessed_token_count,
            //     )
            // }
            Ok(None)
        }
        kalosm_sample::ParseResult::Finished { result, .. } => Ok(Some(result)),
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
