use anyhow::{Error as E, Result};

use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;

use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
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

    pub async fn _infer(
        &mut self,
        inference_settings: &InferenceSettings,
        out: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<()> {
        let InferenceSettings {
            prompt,
            temperature,
            top_p,
            seed,
            sample_len,
            repeat_penalty,
            repeat_last_n,
            ..
        } = inference_settings;

        let mut logits_processor = LogitsProcessor::new(*seed, *temperature, *top_p);
        let mut tokens = self
            .tokenizer
            .encode(&**prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let mut new_tokens = vec![];
        let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };
        for index in 0..*sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if *repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(*repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    *repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = logits_processor.sample(&logits)?;
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
