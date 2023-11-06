use std::{
    fmt::{Debug, Display, Formatter},
    sync::{Arc, Mutex},
};

use kalosm_sample::{ParseResult, Parser, Tokenizer};
use llm_samplers::types::{HasSamplerResources, Sampler, SamplerError};
use rustc_hash::FxHashMap;

use crate::SyncModel;

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_structured<M: SyncModel, P: Parser>(
    prompt: impl Display,
    llm: &mut M,
    tokenizer: &Arc<dyn Tokenizer + Send + Sync>,
    parser: P,
    mut parser_state: P::PartialState,
    mut sampler: Arc<Mutex<dyn Sampler<u32, f32>>>,
    mut post_filter_sampler: Arc<Mutex<dyn Sampler<u32, f32>>>,
    stream: tokio::sync::mpsc::UnboundedSender<String>,
) -> anyhow::Result<P::Output> {
    let prompt_text = prompt.to_string();
    let mut tokens = tokenizer.encode(&prompt_text)?;
    let mut session = llm.new_session()?;
    let mut unprocessed_token_count = tokens.len();
    let mut rng = rand::thread_rng();

    loop {
        let mut logits = llm.feed_tokens(
            &mut session,
            &tokens[tokens.len() - unprocessed_token_count..],
        )?;
        let resources = &mut SamplerResources {
            previous_tokens: &tokens,
            rng: &mut rng,
        };
        let sampled = sampler.sample(resources, &mut logits)?;
        let mut state_map = FxHashMap::default();
        for logit in sampled.iter_mut() {
            let new_text = tokenizer.decode(&[logit.token_id])?;
            if new_text.is_empty() || logit.logit == 0.0 {
                logit.logit = 0.0;
                continue;
            }
            if let Ok(result) = parser.parse(&parser_state, new_text.as_bytes()) {
                let result = result.without_remaining();
                state_map.insert(logit.token_id, (new_text.to_string(), result));
            } else {
                logit.logit = 0.0;
            }
        }
        if state_map.is_empty() {
            return Err(anyhow::anyhow!("No valid tokens found"));
        }
        let token_id = post_filter_sampler
            .sample_token(resources, &mut logits)?
            .ok_or(anyhow::anyhow!("No valid tokens found"))?;
        let (token, result) = state_map
            .remove(&token_id)
            .ok_or(anyhow::anyhow!("Token {} not found in state map", token_id))?;

        stream
            .send(token.clone())
            .map_err(|_| anyhow::anyhow!("Failed to send token to stream: {}", token))?;
        unprocessed_token_count = 1;
        tokens.push(token_id);
        if let Some(result) = update_state(
            &parser,
            &mut parser_state,
            result,
            tokenizer,
            &mut tokens,
            &stream,
            &mut unprocessed_token_count,
        )? {
            return Ok(result);
        }
    }
}

fn update_state<P: Parser>(
    parser: &P,
    parser_state: &mut P::PartialState,
    result: ParseResult<P::PartialState, P::Output>,
    tokenizer: &Arc<dyn Tokenizer + Send + Sync>,
    tokens: &mut Vec<u32>,
    stream: &tokio::sync::mpsc::UnboundedSender<String>,
    unprocessed_token_count: &mut usize,
) -> anyhow::Result<Option<P::Output>> {
    match result {
        kalosm_sample::ParseResult::Incomplete {
            new_state,
            required_next,
        } => {
            *parser_state = new_state;
            if required_next.is_empty() {
                Ok(None)
            } else {
                let result = parser
                    .parse(parser_state, required_next.as_bytes())
                    .unwrap_or_else(|_| unreachable!("Required next should always be valid"));
                let extra_tokens = tokenizer.encode(&required_next)?;
                *unprocessed_token_count += extra_tokens.len();
                tokens.extend(extra_tokens);
                stream
                    .send(required_next.to_string())
                    .map_err(|_| anyhow::anyhow!("Failed to send token to stream"))?;
                update_state(
                    parser,
                    parser_state,
                    result,
                    tokenizer,
                    tokens,
                    stream,
                    unprocessed_token_count,
                )
            }
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
    type TokenId = u32;

    fn with_rng_mut(
        &mut self,
        fun: &mut dyn FnMut(&mut dyn rand::RngCore),
    ) -> Result<(), SamplerError> {
        fun(self.rng);
        Ok(())
    }

    fn with_last_tokens(&self, fun: &mut dyn FnMut(&[Self::TokenId])) -> Result<(), SamplerError> {
        fun(self.previous_tokens);
        Ok(())
    }
}
