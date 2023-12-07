use anyhow::{Error as E, Result};
use kalosm_language_model::Session;
use kalosm_language_model::SyncModel;
use kalosm_language_model::SyncModelExt;
use llm_samplers::prelude::*;
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

impl Session for PhiSession {
    fn save_to(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let tensors = self.get_tensor_map();
        Ok(candle_core::safetensors::save(&tensors, path)?)
    }

    fn load_from(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self>
    where
        Self: std::marker::Sized,
    {
        let device = Device::cuda_if_available(0)?;
        let tensors = candle_core::safetensors::load(path, &device)?;

        Ok(Self::from_tensor_map(tensors))
    }
}

impl PhiSession {
    /// Export the current cache tensor map.
    pub fn get_tensor_map(&self) -> HashMap<String, Tensor> {
        let mut map = self.cache.get_tensor_map();
        map.insert(
            "current_tokens".to_string(),
            Tensor::from_iter(
                self.current_tokens.iter().copied(),
                self.cache.blocks[0].0.as_ref().unwrap().key.device(),
            )
            .unwrap(),
        );
        map
    }

    /// Import a cache tensor map.
    pub fn set_tensor_map(&mut self, map: HashMap<String, Tensor>) {
        self.cache = PhiCache::from_tensor_map(map);
    }

    /// Create a cache from a tensor map. This can be used to load a cache from disk.
    pub fn from_tensor_map(map: HashMap<String, Tensor>) -> Self {
        let current_tokens = map.get("current_tokens").unwrap().to_vec1().unwrap();
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

    fn feed_text(&mut self, session: &mut Self::Session, prompt: &str) -> anyhow::Result<Logits> {
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
            None,
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
        top_k: Option<usize>,
    ) -> anyhow::Result<Logits> {
        if tokens.is_empty() {
            return Err(anyhow::anyhow!("Cannot run model on empty input"));
        }

        if tokens.len() > 4096 {
            tokens = &tokens[tokens.len() - 4096..];
        }

        let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
        let logits = model.forward(&input, cache)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let logits: Vec<f32> = logits.to_vec1()?;
        match top_k {
            Some(top_k) => Ok(Logits::try_from_iter_top_k(logits, top_k)?),
            None => Ok(Logits::try_from_iter(logits)?),
        }
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
        sampler: std::sync::Arc<std::sync::Mutex<dyn llm_samplers::prelude::Sampler>>,
        out: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<()> {
        let InferenceSettings {
            prompt,
            sample_len,
            stop_on,
        } = settings;

        let mut session = self.new_session()?;

        self.stream_text_with_sampler(
            &mut session,
            prompt.as_str(),
            Some(sample_len as u32),
            stop_on.as_deref(),
            sampler,
            |token| {
                out.send(token)
                    .map_err(|_| anyhow::anyhow!("Failed to send token to output channel"))
                    .map(|_| kalosm_language_model::ModelFeedback::Continue)
            },
        )?;

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
