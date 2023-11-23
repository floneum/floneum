use crate::{
    raw::Model,
    session::{LlamaCache, LlamaSession},
};
use anyhow::{Error as E, Result};
use llm_samplers::{
    prelude::Logits,
    types::{HasSamplerResources, Sampler, SamplerError},
};
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use kalosm_language_model::SyncModel;
use rand::SeedableRng;
use tokenizers::Tokenizer;

use crate::InferenceSettings;

/// The inner, synchronous Llama model.
pub struct LlamaModel {
    model: Model,
    device: Device,
    tokenizer: Tokenizer,
    cache: LlamaCache,
}

impl SyncModel for LlamaModel {
    type Session = LlamaSession;

    fn new_session(&self) -> anyhow::Result<Self::Session> {
        let mut cache = self.cache.clone();
        cache.clear();
        Ok(Self::Session {
            cache,
            current_tokens: Vec::new(),
        })
    }

    fn feed_text(&mut self, session: &mut Self::Session, prompt: &str) -> anyhow::Result<Logits> {
        let encoded = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        let tokens = encoded.get_ids();
        self.feed_tokens(session, tokens)
    }

    fn feed_tokens(
        &mut self,
        session: &mut Self::Session,
        tokens: &[u32],
    ) -> anyhow::Result<Logits> {
        let first_token = session.current_tokens.is_empty();

        if first_token {
            session.current_tokens.extend(tokens);
            Self::forward(
                &mut self.model,
                &self.device,
                &session.current_tokens,
                0,
                Some(&mut session.cache),
                None,
            )
        } else {
            for tid in tokens.iter().copied().take(tokens.len() - 1) {
                let seq_len_offset = session.current_tokens.len();
                session.current_tokens.push(tid);
                Self::forward(
                    &mut self.model,
                    &self.device,
                    &[tid],
                    seq_len_offset,
                    Some(&mut session.cache),
                    None,
                )?;
            }
            let tid = *tokens.last().unwrap();
            let seq_len_offset = session.current_tokens.len();
            session.current_tokens.push(tid);
            Self::forward(
                &mut self.model,
                &self.device,
                &[tid],
                seq_len_offset,
                Some(&mut session.cache),
                None,
            )
        }
    }

    fn stop_token(&self) -> anyhow::Result<u32> {
        let eos_token = match self.tokenizer.get_vocab(true).get("</s>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the </s> token"),
        };
        Ok(eos_token)
    }

    fn tokenizer(&self) -> std::sync::Arc<dyn kalosm_sample::Tokenizer + Send + Sync> {
        Arc::new(self.tokenizer.clone())
            as std::sync::Arc<dyn kalosm_sample::Tokenizer + Send + Sync>
    }
}

impl LlamaModel {
    fn forward(
        model: &mut Model,
        device: &Device,
        tokens: &[u32],
        seqlen_offset: usize,
        cache: Option<&mut LlamaCache>,
        top_k: Option<usize>,
    ) -> anyhow::Result<Logits> {
        if tokens.is_empty() {
            return Err(anyhow::anyhow!("Cannot run model on empty input"));
        }

        let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
        let logits = model.forward(&input, seqlen_offset, cache)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits: Vec<f32> = logits.to_vec1()?;
        match top_k {
            Some(top_k) => Ok(Logits::try_from_iter_top_k(logits, top_k)?),
            None => Ok(Logits::try_from_iter(logits)?),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        model: Model,
        tokenizer: Tokenizer,
        device: Device,
        cache: LlamaCache,
    ) -> Self {
        Self {
            cache,
            model,
            device,
            tokenizer,
        }
    }

    pub(crate) fn _infer(
        &mut self,
        settings: InferenceSettings,
        mut sampler: std::sync::Arc<std::sync::Mutex<dyn llm_samplers::prelude::Sampler>>,
        out: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<()> {
        let InferenceSettings {
            prompt,
            sample_len,
            seed,
            stop_on,
        } = settings;

        self.cache.clear();

        let mut tokens = self
            .tokenizer
            .encode(&*prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let eos_token = self.stop_token()?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut text = String::new();
        let mut prev_index = tokens.len();
        let mut current_index = tokens.len();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let logits = Self::forward(
                &mut self.model,
                &self.device,
                ctxt,
                start_pos,
                Some(&mut self.cache),
                Some(1000),
            )?;
            let next_token = sample_token(
                &mut sampler,
                &mut rng,
                &tokens,
                logits,
                stop_on.as_deref(),
                &self.tokenizer,
            )?;
            if next_token == eos_token {
                break;
            }
            let prev_text = if tokens.is_empty() {
                String::new()
            } else {
                let tokens = &tokens[prev_index..current_index];
                self.tokenizer.decode(tokens, true).map_err(E::msg)?
            };
            tokens.push(next_token);
            let token_text = self
                .tokenizer
                .decode(&tokens[prev_index..], true)
                .map_err(E::msg)?;
            let token = if token_text.len() > prev_text.len()
                && token_text.chars().last().unwrap().is_ascii()
            {
                let text = token_text.split_at(prev_text.len());
                prev_index = current_index;
                current_index = tokens.len();
                text.1.to_string()
            } else {
                continue;
            };

            let mut should_stop = false;
            // We only need to keep as many bytes as the stop_on string
            if let Some(stop_on) = &stop_on {
                text.push_str(&token);
                should_stop = text.ends_with(stop_on);

                if text.len() > stop_on.len() {
                    text = text[text.len() - stop_on.len()..].to_string();
                }
            }
            out.send(token).unwrap();
            if should_stop {
                break;
            }
        }

        Ok(())
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

pub fn sample_token(
    sampler: &mut impl Sampler,
    rng: &mut impl rand::Rng,
    previous_tokens: &[u32],
    mut last_logits: Logits,
    stop_on: Option<&str>,
    tokenizer: &Tokenizer,
) -> anyhow::Result<u32> {
    let mut end_tokens = String::new();
    // grab as many characters as the stop_on string has from the end of the previous tokens
    if let Some(stop_on) = stop_on {
        let required_len = stop_on.len();
        let mut previous_token_iter = previous_tokens.iter().rev();
        while end_tokens.len() < required_len {
            match previous_token_iter.next() {
                Some(token) => {
                    end_tokens = tokenizer.decode(&[*token], true).map_err(E::msg)? + &end_tokens;
                }
                None => {
                    break;
                }
            }
        }
    }
    for logit in last_logits.iter_mut() {
        let tid = logit.token_id;
        if let Some(stop_on) = stop_on {
            let token = tokenizer.decode(&[tid], true).unwrap();
            let combined = end_tokens.clone() + &token;
            if combined.contains(stop_on) && !combined.ends_with(stop_on) {
                // if the token contains a stop_on token, but not the end of the string, set the probability to 0
                logit.prob = 0.0;
            }
        }
    }
    last_logits
        .sample_token(
            &mut SamplerResources {
                previous_tokens,
                rng,
            },
            sampler,
        )?
        .ok_or_else(|| anyhow::anyhow!("No token sampled"))
}
