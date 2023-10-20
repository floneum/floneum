use anyhow::{Error as E, Result};
use floneumin_language_model::SyncModel;
use llm_samplers::prelude::*;
use rand::SeedableRng;
use std::fmt::Debug;
use std::fmt::Formatter;

use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;

use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

use crate::InferenceSettings;

/// The inner, synchronous Phi-1.5 model.
pub struct PhiModel {
    model: QMixFormer,
    device: Device,
    tokenizer: Tokenizer,
}

impl SyncModel for PhiModel {
    fn run(&mut self, prompt: &str) -> anyhow::Result<Logits<u32, f32>> {
        let tokens = self
            .tokenizer
            .encode(&*prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        self.forward(&tokens, 0)
    }

    fn stop_token(&self) -> anyhow::Result<u32> {
        let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };
        Ok(eos_token)
    }
}

impl PhiModel {
    fn forward(&mut self, mut tokens: &[u32], index: usize) -> anyhow::Result<Logits<u32, f32>> {
        if tokens.len() > 4096 {
            tokens = &tokens[tokens.len() - 4096..];
        }
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let logits: Vec<f32> = logits.to_vec1()?;
        Ok(Logits::try_from_iter(logits)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(model: QMixFormer, tokenizer: Tokenizer, device: Device) -> Self {
        Self {
            model,
            device,
            tokenizer,
        }
    }

    pub(crate) fn _infer(
        &mut self,
        settings: InferenceSettings,
        mut sampler: std::sync::Arc<std::sync::Mutex<dyn llm_samplers::prelude::Sampler<u32, f32>>>,
        out: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<()> {
        let InferenceSettings {
            prompt,
            sample_len,
            seed,
            stop_on,
        } = settings;

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
            let logits = self.forward(&tokens, index)?;
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

pub fn sample_token(
    sampler: &mut impl Sampler<u32, f32>,
    rng: &mut impl rand::Rng,
    previous_tokens: &[u32],
    mut last_logits: Logits<u32, f32>,
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
            let token = tokenizer.decode(&[tid as u32], true).unwrap();
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
