use anyhow::{Error as E, Result};

use candle_transformers::models::quantized_mistral::Model;

use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

use crate::InferenceSettings;

pub(crate) struct MistralInner {
    model: Model,
    device: Device,
    tokenizer: Tokenizer,
}

impl MistralInner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(model: Model, tokenizer: Tokenizer, device: Device) -> Self {
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
        for index in 0..*sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
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
            let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
            println!("{}", token);
            out.send(token).unwrap();
        }

        Ok(())
    }
}
