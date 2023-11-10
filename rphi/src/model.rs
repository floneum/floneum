use anyhow::{Error as E, Result};
use kalosm_language_model::SyncModel;
use llm_samplers::prelude::*;
use rand::SeedableRng;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use crate::raw::MixFormerSequentialForCausalLM as QMixFormer;
use crate::raw::PhiCache;

use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

use crate::InferenceSettings;

/// A Phi-1.5 session.
pub struct PhiSession {
    cache: PhiCache,
    current_tokens: Vec<u32>,
}

impl PhiSession {
    /// Export the current cache tensor map.
    pub fn get_tensor_map(&self) -> HashMap<String, Tensor> {
        self.cache.get_tensor_map()
    }

    /// Import a cache tensor map.
    pub fn set_tensor_map(&mut self, map: HashMap<String, Tensor>) {
        self.cache = PhiCache::from_tensor_map(map);
    }

    /// Create a cache from a tensor map. This can be used to load a cache from disk.
    pub fn from_tensor_map(map: HashMap<String, Tensor>, current_tokens: Vec<u32>) -> Self {
        Self {
            cache: PhiCache::from_tensor_map(map),
            current_tokens,
        }
    }

    /// Get the current tokens.
    pub fn get_current_tokens(&self) -> &[u32] {
        &self.current_tokens
    }
}

/// The inner, synchronous Phi-1.5 model.
pub struct PhiModel {
    cache: PhiCache,
    model: QMixFormer,
    device: Device,
    tokenizer: Tokenizer,
}

impl SyncModel for PhiModel {
    type Session = PhiSession;

    fn new_session(&self) -> anyhow::Result<Self::Session> {
        let mut cache = self.cache.clone();
        cache.clear();
        Ok(PhiSession {
            cache,
            current_tokens: vec![],
        })
    }

    fn feed_text(
        &mut self,
        session: &mut Self::Session,
        prompt: &str,
    ) -> anyhow::Result<Logits> {
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        self.feed_tokens(session, &tokens)
    }

    fn feed_tokens(
        &mut self,
        session: &mut Self::Session,
        tokens: &[u32],
    ) -> anyhow::Result<Logits> {
        session.current_tokens.extend(tokens.iter().copied());

        Self::forward(
            &mut self.model,
            &self.device,
            tokens,
            Some(&mut session.cache),
        )
    }

    fn stop_token(&self) -> anyhow::Result<u32> {
        let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };
        Ok(eos_token)
    }

    fn tokenizer(&self) -> Arc<dyn kalosm_sample::Tokenizer + Send + Sync> {
        Arc::new(self.tokenizer.clone())
    }
}

impl PhiModel {
    fn forward(
        model: &mut QMixFormer,
        device: &Device,
        mut tokens: &[u32],
        cache: Option<&mut PhiCache>,
    ) -> anyhow::Result<Logits> {
        if tokens.is_empty() {
            return Err(anyhow::anyhow!("Cannot run model on empty input"));
        }

        if tokens.len() > 4096 {
            tokens = &tokens[tokens.len() - 4096..];
        }

        let input = Tensor::new(tokens, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, cache)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let logits: Vec<f32> = logits.to_vec1()?;
        Ok(Logits::try_from_iter(logits)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        model: QMixFormer,
        tokenizer: Tokenizer,
        device: Device,
        cache: PhiCache,
    ) -> Self {
        Self {
            model,
            device,
            tokenizer,
            cache,
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

        let mut new_tokens = vec![];
        let eos_token = self.stop_token()?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut text = String::new();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let logits = Self::forward(&mut self.model, &self.device, ctxt, Some(&mut self.cache))?;
            let next_token = sample_token(
                &mut sampler,
                &mut rng,
                &tokens,
                logits,
                stop_on.as_deref(),
                &self.tokenizer,
            )?;
            tokens.push(next_token);
            new_tokens.push(next_token);
            if next_token == eos_token {
                break;
            }
            let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
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
