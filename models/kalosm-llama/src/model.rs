use crate::{
    raw::Model,
    session::{LlamaCache, LlamaSession},
};
use anyhow::{Error as E, Result};
use kalosm_language_model::SyncModelExt;
use llm_samplers::prelude::Logits;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use kalosm_language_model::SyncModel;
use tokenizers::Tokenizer;

use crate::InferenceSettings;

/// The inner, synchronous Llama model.
pub struct LlamaModel {
    model: Model,
    device: Device,
    tokenizer: Arc<Tokenizer>,
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

    fn feed_text(&self, session: &mut Self::Session, prompt: &str) -> anyhow::Result<Logits> {
        let encoded = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        let tokens = encoded.get_ids();
        self.feed_tokens(session, tokens)
    }

    fn feed_tokens(&self, session: &mut Self::Session, tokens: &[u32]) -> anyhow::Result<Logits> {
        let first_token = session.current_tokens.is_empty();

        if first_token {
            session.current_tokens.extend(tokens);
            Self::forward(
                &self.model,
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
                    &self.model,
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
                &self.model,
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
        self.tokenizer.clone() as std::sync::Arc<dyn kalosm_sample::Tokenizer + Send + Sync>
    }
}

impl LlamaModel {
    fn forward(
        model: &Model,
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
        tokenizer: Arc<Tokenizer>,
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
