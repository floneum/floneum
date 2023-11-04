use std::{
    fmt::{Debug, Display, Formatter},
    sync::{Arc, Mutex},
};

use kalosm_sample::{Parser, Tokenizer};
use llm_samplers::types::{HasSamplerResources, Sampler, SamplerError};
use rand::Rng;

use crate::SyncModel;

pub(crate) fn generate_structured<M: SyncModel, P: Parser>(
    prompt: impl Display,
    llm: &mut M,
    tokenizer: &Arc<dyn Tokenizer + Send + Sync>,
    parser: P,
    mut parser_state: P::PartialState,
    mut sampler: Arc<Mutex<dyn Sampler<u32, f32>>>,
    stream: tokio::sync::mpsc::UnboundedSender<String>,
) -> anyhow::Result<P::Output> {
    let prompt_text = prompt.to_string();
    let mut tokens = tokenizer.encode(&prompt_text).unwrap();
    let mut session = llm.new_session().unwrap();
    let mut unprocessed_token_count = tokens.len();
    let mut rng = rand::thread_rng();

    loop {
        let mut logits = llm
            .feed_tokens(
                &mut session,
                &tokens[tokens.len() - unprocessed_token_count..],
            )
            .unwrap();
        logits.ensure_sorted().unwrap();
        let min_prob = logits.last().unwrap().logit;
        for logit in logits.iter_mut() {
            logit.logit = logit.logit - min_prob;
            debug_assert!(logit.logit >= 0.0)
        }
        let resources = &mut SamplerResources {
            previous_tokens: &tokens,
            rng: &mut rng,
        };
        let sampled = sampler.sample(resources, &mut logits)?;
        let mut choices = Vec::with_capacity(sampled.len());
        for logit in logits.iter() {
            let new_text = tokenizer.decode(&[logit.token_id]).unwrap();
            if new_text.is_empty() || logit.logit == 0.0 {
                continue;
            }
            if let Ok(result) = parser.parse(&parser_state, new_text.as_bytes()) {
                let result = result.without_remaining();
                let prob = logit.logit;
                choices.push((logit.token_id, new_text.to_string(), result, prob));
            }
        }
        if choices.is_empty() {
            return Err(anyhow::anyhow!("No valid tokens found"));
        }
        let total = choices.iter().map(|(_, _, _, prob)| prob).sum::<f32>();
        let mut rng = rand::thread_rng();
        let random_choice = if total == 0.0 {
            0.0
        } else {
            rng.gen_range(0.0..total)
        };
        let mut best_token = None;

        let mut total = 0.0;
        for (token_id, token, result, prob) in choices {
            total += prob;
            if total >= random_choice {
                best_token = Some((token_id, token, result));
                break;
            }
        }
        let (token_id, token, result) = best_token.unwrap();

        stream
            .send(token.clone())
            .map_err(|_| anyhow::anyhow!("Failed to send token to stream: {}", token))?;
        unprocessed_token_count = 1;
        tokens.push(token_id);
        match result {
            kalosm_sample::ParseResult::Incomplete(new_state) => {
                parser_state = new_state;
            }
            kalosm_sample::ParseResult::Finished { result, .. } => {
                return Ok(result);
            }
        }
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
