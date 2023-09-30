use anyhow::{Error as E, Result};
use llm_samplers::prelude::*;
use rand::SeedableRng;
use std::fmt::Debug;
use std::fmt::Formatter;

use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;

use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

use crate::InferenceSettings;

pub(crate) struct PhiInner {
    model: QMixFormer,
    device: Device,
    tokenizer: Tokenizer,
}

impl PhiInner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(model: QMixFormer, tokenizer: Tokenizer, device: Device) -> Self {
        Self {
            model,
            device: device,
            tokenizer,
        }
    }

    pub fn _infer(
        &mut self,
        settings: InferenceSettings,
        mut sampler: std::sync::Arc<std::sync::Mutex<dyn llm_samplers::prelude::Sampler<u32, f32>>>,
        out: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<()> {
        let InferenceSettings {
            prompt,
            sample_len,
            seed,
        } = settings;

        let mut tokens = self
            .tokenizer
            .encode(&*prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let mut new_tokens = vec![];
        let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        for index in 0..sample_len as usize {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits: Vec<f32> = logits.to_vec1()?;
            let next_token = sample_token(&mut sampler, &mut rng, &tokens, logits)?;
            tokens.push(next_token);
            new_tokens.push(next_token);
            if next_token == eos_token {
                break;
            }
            let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
            out.send(token).unwrap();
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
    last_logits: impl IntoIterator<Item = f32>,
) -> anyhow::Result<u32> {
    Ok(Logits::try_from_iter(last_logits.into_iter())?
        .sample_token(
            &mut SamplerResources {
                previous_tokens,
                rng,
            },
            sampler,
        )?
        .ok_or_else(|| anyhow::anyhow!("No token sampled"))?)
}
