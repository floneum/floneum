use crate::raw::cache::LlamaCache;
use crate::{raw::Model, session::LlamaSession};
use kalosm_common::*;
use kalosm_language_model::{SyncModelExt, UnstructuredTextGenerationError};
use std::sync::Arc;

use candle_core::{
    quantized::{ggml_file, gguf_file},
    DType, Device,
};
use kalosm_language_model::SyncModel;
use tokenizers::Tokenizer;

use crate::{InferenceSettings, LlamaSourceError};

/// An error that can occur when running a [`LlamaModel`].
#[derive(Debug, thiserror::Error)]
pub enum LlamaModelError {
    /// An error from candle while running the model.
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    /// An error from tokenizers while running the model.
    #[error("Tokenizer error: {0}")]
    Tokenizer(tokenizers::Error),
    /// No stop token was found.
    #[error("No stop token was found")]
    NoStopToken,
}

/// The inner, synchronous Llama model.
pub struct LlamaModel {
    model: Model,
    device: Device,
    tokenizer: Arc<Tokenizer>,
    cache: LlamaCache,
}

impl SyncModel for LlamaModel {
    type Session = LlamaSession;
    type Error = LlamaModelError;

    fn new_session(&self) -> Result<Self::Session, Self::Error> {
        let cache = self.cache.clone();
        Ok(Self::Session { cache })
    }

    fn feed_text(
        &self,
        session: &mut Self::Session,
        prompt: &str,
        logits: &mut Vec<f32>,
    ) -> Result<(), Self::Error> {
        let encoded = self
            .tokenizer
            .encode(prompt, false)
            .map_err(LlamaModelError::Tokenizer)?;
        let tokens = encoded.get_ids();
        self.feed_tokens(session, tokens, logits)
    }

    fn feed_tokens(
        &self,
        session: &mut Self::Session,
        tokens: &[u32],
        logits: &mut Vec<f32>,
    ) -> Result<(), Self::Error> {
        Self::forward(
            &self.model,
            &self.device,
            tokens,
            Some(&mut session.cache),
            logits,
        )?;
        Ok(())
    }

    fn stop_token(&self) -> Result<u32, Self::Error> {
        let vocab = self.tokenizer.get_vocab(true);
        let eos_token = match vocab.get("</s>").or(vocab.get("<|end_of_text|>")) {
            Some(token) => *token,
            None => return Err(LlamaModelError::NoStopToken),
        };
        Ok(eos_token)
    }

    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }
}

impl LlamaModel {
    fn forward(
        model: &Model,
        device: &Device,
        tokens: &[u32],
        cache: Option<&mut LlamaCache>,
        logits_vec: &mut Vec<f32>,
    ) -> candle_core::Result<()> {
        if tokens.is_empty() {
            candle_core::bail!("Cannot run model on empty input");
        }

        let logits = model.forward(tokens, device, cache)?;

        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        copy_tensor_into_vec(&logits, logits_vec)?;

        Ok(())
    }

    /// Create a new sync Llama model from a builder.
    pub async fn from_builder(
        builder: crate::LlamaBuilder,
        mut handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Self, LlamaSourceError> {
        let device = builder.get_device()?;

        let tokenizer_source = format!("Tokenizer ({})", builder.source.tokenizer);
        let mut create_progress = ModelLoadingProgress::downloading_progress(tokenizer_source);
        let tokenizer = builder
            .source
            .tokenizer(|progress| handler(create_progress(progress)))
            .await?;

        let source = format!("Model ({})", builder.source.model);
        let mut create_progress = ModelLoadingProgress::downloading_progress(source);
        let filename = builder
            .source
            .model(|progress| handler(create_progress(progress)))
            .await?;
        let mut file = std::fs::File::open(&filename)
            .expect("The path returned by LlamaSource::model should be valid");
        let model = match filename.extension().and_then(|v| v.to_str()) {
            Some("gguf") => {
                let model = gguf_file::Content::read(&mut file)?;
                Model::from_gguf(model, &mut file, &device)?
            }
            Some("ggml" | "bin") | Some(_) | None => {
                let model = ggml_file::Content::read(&mut file, &device)?;
                let gqa = builder.source.group_query_attention;
                Model::from_ggml(model, gqa as usize, &device)?
            }
        };

        let cache = LlamaCache::new(&model.config);
        Ok(Self {
            model,
            tokenizer: Arc::new(tokenizer),
            device,
            cache,
        })
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
    ) -> Result<(), UnstructuredTextGenerationError<LlamaModelError>> {
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
            |token| match out.send(token) {
                Ok(_) => Ok(kalosm_language_model::ModelFeedback::Continue),
                Err(_) => Ok(kalosm_language_model::ModelFeedback::Stop),
            },
        )?;

        Ok(())
    }
}

/// An extension trait for sync models.
pub trait SyncModelExt: SyncModel {
    /// Generate new text with the given prompt that conforms to the given parser.
    #[allow(clippy::too_many_arguments)]
    fn generate_structured<P: Parser>(
        &self,
        session: &mut Self::Session,
        prompt: impl Display,
        parser: P,
        parser_state: P::PartialState,
        sampler: Arc<Mutex<dyn Sampler>>,
        on_token: impl FnMut(String) -> Result<(), Self::Error>,
        top_k: Option<usize>,
    ) -> Result<P::Output, StructuredTextGenerationError<Self::Error>> {
        generate_structured(
            prompt,
            self,
            session,
            parser,
            parser_state,
            sampler,
            on_token,
            top_k,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Stream text, calling the on_token callback every time a new token is generated. For some models, this could be used to implement [`Model::stream_text_with_sampler`].
    fn stream_text_with_sampler(
        &self,
        session: &mut Self::Session,
        prompt: &str,
        max_tokens: Option<u32>,
        stop_on: Option<&str>,
        mut sampler: Arc<Mutex<dyn Sampler>>,
        mut on_token: impl FnMut(String) -> Result<ModelFeedback, Self::Error>,
    ) -> Result<(), UnstructuredTextGenerationError<Self::Error>> {
        let tokens = self
            .tokenizer()
            .encode(prompt, false)
            .map_err(UnstructuredTextGenerationError::TokenizationError)?;
        let tokens = tokens.get_ids();
        let mut text_stream = TokenOutputStream::new(self.tokenizer());
        for &token in tokens {
            text_stream
                .next_token(token)
                .map_err(UnstructuredTextGenerationError::TokenOutputStreamError)?;
        }

        let mut logit_probs = Vec::new();
        self.feed_tokens(session, tokens, &mut logit_probs)?;
        let mut logits = Logits::try_from_iter_top_k(logit_probs, 512)
            .expect("model output should be valid logits");
        let mut tokens_generated = 0;
        // This stores a buffer of text that has been generated to check against the stop_on string. It should never be longer than the stop_on string.
        let mut queued_text_matching_stop_on = String::new();
        let stop_on_lowercase = stop_on.map(|s| s.to_lowercase());
        let stop_on_lowercase = stop_on_lowercase.as_deref();
        let stop_token = self.stop_token()?;
        let mut logit_probs = Vec::new();

        'generate: loop {
            let new_token = text_stream
                .sample_token(&mut sampler, logits, stop_on)
                .map_err(UnstructuredTextGenerationError::TokenOutputStreamError)?;
            if new_token == stop_token {
                tracing::trace!("Stopping on stop token");
                break;
            }
            if let Some(mut new_text) = text_stream
                .next_token(new_token)
                .map_err(UnstructuredTextGenerationError::TokenOutputStreamError)?
            {
                if let Some(stop_on) = stop_on_lowercase {
                    let lowercase = new_text.to_lowercase();

                    // Check if the string ends with the start of the stop_on string
                    let mut before_stop_on = None;
                    let remaining_stop_on = stop_on
                        .strip_prefix(&queued_text_matching_stop_on)
                        .unwrap_or(stop_on);

                    // If the remaining stop_on string is empty, we have found a match
                    if remaining_stop_on.is_empty() {
                        break;
                    }

                    for (i, _) in lowercase.char_indices() {
                        let end_of_new_text = &lowercase[i..];
                        if end_of_new_text.is_empty() {
                            break;
                        }

                        // Check if we have matched all of the stop_on string
                        if end_of_new_text.starts_with(remaining_stop_on) {
                            queued_text_matching_stop_on += end_of_new_text;
                            break 'generate;
                        }

                        // Check if the string ends with the start of the stop_on string
                        if remaining_stop_on.starts_with(end_of_new_text) {
                            before_stop_on = Some(lowercase[..i].to_string());
                            queued_text_matching_stop_on += end_of_new_text;
                            break;
                        }
                    }

                    match before_stop_on {
                        Some(before_stop_on) => {
                            if let ModelFeedback::Stop = on_token(before_stop_on)? {
                                break;
                            }
                        }
                        None => {
                            new_text =
                                std::mem::take(&mut queued_text_matching_stop_on) + &new_text;
                            if let ModelFeedback::Stop = on_token(new_text)? {
                                break;
                            }
                        }
                    }
                } else if let ModelFeedback::Stop = on_token(new_text)? {
                    break;
                }
            }
            tokens_generated += 1;
            if let Some(max_tokens) = max_tokens {
                if tokens_generated >= max_tokens {
                    break;
                }
            }
            self.feed_tokens(session, &[new_token], &mut logit_probs)?;
            logits = Logits::try_from_iter_top_k(logit_probs.iter().copied(), 512)
                .expect("model output should be valid logits");
        }

        // Flush the queued text
        if let Some(stop_string) = stop_on_lowercase {
            if !queued_text_matching_stop_on.starts_with(stop_string) {
                on_token(queued_text_matching_stop_on)?;
            }
        }

        Ok(())
    }
}
