use crate::gguf_tokenizer::get_pre_tokenizer;
use crate::raw::cache::LlamaCache;
use crate::raw::Model;
use crate::token_stream::TokenOutputStream;
use crate::token_stream::TokenOutputStreamError;
use crate::LlamaConfigJson;
use crate::LlamaSession;
use kalosm_common::*;
use kalosm_model_types::ModelLoadingProgress;
use llm_samplers::types::Logits;
use rustc_hash::FxHashSet;
use serde::de::Error;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{
    quantized::{ggml_file, gguf_file},
    DType, Device,
};
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

    /// An error while sampling tokens.
    #[error("Sampler error: {0}")]
    SamplerError(Box<dyn std::error::Error + Send + Sync>),

    /// A streaming detokenization error.
    #[error("Token output stream error: {0}")]
    TokenOutputStreamError(TokenOutputStreamError),

    /// An error while writing to the session cache.
    #[error("Session cache error: {0}")]
    Session(String),

    /// No valid tokens were sampled during structured generation
    #[error("No valid tokens were sampled")]
    NoValidTokens,

    /// The model has already stopped.
    #[error("Model stopped")]
    ModelStopped,

    /// No chat template was provided
    #[error("No chat template was provided")]
    NoChatTemplate,

    /// Error running the chat template
    #[error("Error running the chat template: {0}")]
    ChatTemplateError(#[from] minijinja::Error),
}

/// The inner, synchronous Llama model.
pub struct LlamaModel {
    pub(crate) model: Model,
    pub(crate) device: Device,
    pub tokenizer: Arc<Tokenizer>,
    pub(crate) merges: FxHashSet<[u32; 2]>,
}

impl LlamaModel {
    pub(crate) fn forward(
        model: &Model,
        device: &Device,
        tokens: &[u32],
        cache: Option<&mut LlamaCache>,
        logits_vec: &mut Vec<f32>,
        #[allow(unused)] tokenizer: &Tokenizer,
    ) -> candle_core::Result<()> {
        if tokens.is_empty() {
            candle_core::bail!("Cannot run model on empty input");
        }

        #[cfg(debug_assertions)]
        {
            tracing::trace!(
                "Running model with tokens: {:?}",
                tokenizer.decode(tokens, false)
            );
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

        // Download the model and tokenizer. These are relatively cheep operations that can be run in the async runtime
        let tokenizer_path = match &builder.source.tokenizer {
            Some(tokenizer) => {
                let tokenizer_source = format!("Tokenizer ({})", tokenizer);
                let mut create_progress =
                    ModelLoadingProgress::downloading_progress(tokenizer_source);
                let tokenizer_path = builder
                    .source
                    .cache
                    .get(tokenizer, |progress| handler(create_progress(progress)))
                    .await?;
                Some(tokenizer_path)
            }
            None => None,
        };

        // Download the config file if it exists
        let config_path = match &builder.source.config {
            Some(config) => {
                let config_source = format!("Config ({})", config);
                let mut create_progress = ModelLoadingProgress::downloading_progress(config_source);
                let config_path = builder
                    .source
                    .cache
                    .get(config, |progress| handler(create_progress(progress)))
                    .await?;
                Some(config_path)
            }
            None => None,
        };

        let source = format!("Model ({})", builder.source.model);
        let mut create_progress = ModelLoadingProgress::downloading_progress(source);
        let filename = builder
            .source
            .model(|progress| handler(create_progress(progress)))
            .await?;

        let merges = extract_merges(
            &std::fs::read(tokenizer_path.as_ref().unwrap()).map_err(|err| {
                LlamaSourceError::Tokenizer(serde_json::Error::custom(err).into())
            })?,
        )?;

        // Then actually load the model and tokenizer. This is expensive, so we do it in a blocking task
        let (model, tokenizer) = tokio::task::spawn_blocking({
            let device = device.clone();
            move || {
                let tokenizer = match tokenizer_path {
                    Some(tokenizer_path) => {
                        let tokenizer = Tokenizer::from_file(tokenizer_path)
                            .map_err(LlamaSourceError::Tokenizer)?;
                        Some(tokenizer)
                    }
                    None => None,
                };

                let config = match config_path {
                    Some(config_path) => {
                        let config = std::fs::read_to_string(config_path).map_err(|err| {
                            LlamaSourceError::Config(serde_json::Error::custom(err))
                        })?;
                        let config: LlamaConfigJson =
                            serde_json::from_str(&config).map_err(LlamaSourceError::Config)?;
                        config.rope_scaling
                    }
                    None => None,
                };

                let mut file = std::fs::File::open(&filename)
                    .expect("The path returned by LlamaSource::model should be valid");
                let override_stop_token_string = builder.source.override_stop_token_string;
                match filename.extension().and_then(|v| v.to_str()) {
                    Some("gguf") => {
                        let model = gguf_file::Content::read(&mut file)?;
                        let tokenizer = match tokenizer {
                            Some(tokenizer) => tokenizer,
                            None => {
                                let tokenizer_model = model
                                    .metadata
                                    .get("tokenizer.ggml.model")
                                    .ok_or(LlamaSourceError::NoTokenizer)?
                                    .to_string()
                                    .map_err(|_| LlamaSourceError::NoTokenizer)?;
                                if tokenizer_model != "gpt2" {
                                    return Err(LlamaSourceError::NoTokenizer);
                                }
                                let pre = model
                                    .metadata
                                    .get("tokenizer.ggml.pre")
                                    .ok_or(LlamaSourceError::NoTokenizer)?
                                    .to_string()
                                    .map_err(|_| LlamaSourceError::NoTokenizer)?;
                                let add_bos_token = model
                                    .metadata
                                    .get("tokenizer.ggml.add_bos_token")
                                    .and_then(|v| v.to_bool().ok());
                                let config = get_pre_tokenizer(pre, add_bos_token);

                                let tokens: Result<Vec<_>, _> = model
                                    .metadata
                                    .get("tokenizer.ggml.tokens")
                                    .ok_or(LlamaSourceError::NoTokenizer)?
                                    .to_vec()
                                    .map_err(|_| LlamaSourceError::NoTokenizer)?
                                    .iter()
                                    .map(|v| v.to_string().map(|s| s.to_string()))
                                    .collect();
                                let tokens = tokens.map_err(|_| LlamaSourceError::NoTokenizer)?;
                                let types: Result<Vec<_>, _> = model
                                    .metadata
                                    .get("tokenizer.ggml.token_type")
                                    .ok_or(LlamaSourceError::NoTokenizer)?
                                    .to_vec()
                                    .map_err(|_| LlamaSourceError::NoTokenizer)?
                                    .iter()
                                    .map(|v| {
                                        v.to_i32()
                                            .map(|v| v as u8)
                                            .or_else(|_| v.to_i64().map(|v| v as u8))
                                            .or_else(|_| v.to_i16().map(|v| v as u8))
                                            .or_else(|_| v.to_i8().map(|v| v as u8))
                                            .or_else(|_| v.to_u64().map(|v| v as u8))
                                            .or_else(|_| v.to_u32().map(|v| v as u8))
                                            .or_else(|_| v.to_u16().map(|v| v as u8))
                                            .or_else(|_| v.to_u8())
                                    })
                                    .collect();
                                let types = types.map_err(|_| LlamaSourceError::NoTokenizer)?;
                                let vocab: HashMap<_, _> = tokens
                                    .iter()
                                    .enumerate()
                                    .map(|(id, v)| (v.clone(), id as u32))
                                    .collect();
                                let merges = model
                                    .metadata
                                    .get("tokenizer.ggml.merges")
                                    .ok_or(LlamaSourceError::NoTokenizer)?;
                                let merges: Result<Vec<_>, _> = merges
                                    .to_vec()
                                    .map_err(|_| LlamaSourceError::NoTokenizer)?
                                    .iter()
                                    .map(|v| {
                                        v.to_string()
                                            .map_err(|_| LlamaSourceError::NoTokenizer)
                                            .and_then(|v| {
                                                v.split_once(' ')
                                                    .ok_or(LlamaSourceError::NoTokenizer)
                                            })
                                            .map(|(a, b)| (a.to_string(), b.to_string()))
                                    })
                                    .collect();
                                let merges = merges.map_err(|_| LlamaSourceError::NoTokenizer)?;

                                let eos = model
                                    .metadata
                                    .get("tokenizer.ggml.eos_token_id")
                                    .ok_or(LlamaSourceError::NoTokenizer)?;
                                let eos =
                                    eos.to_u32().map_err(|_| LlamaSourceError::NoTokenizer)?;
                                let eos = &tokens[eos as usize];

                                let bos = model
                                    .metadata
                                    .get("tokenizer.ggml.bos_token_id")
                                    .ok_or(LlamaSourceError::NoTokenizer)?;
                                let bos =
                                    bos.to_u32().map_err(|_| LlamaSourceError::NoTokenizer)?;
                                let bos = &tokens[bos as usize];

                                config
                                    .build(vocab, types, merges, bos, eos)
                                    .map_err(LlamaSourceError::Tokenizer)?
                            }
                        };
                        let model = Model::from_gguf(
                            model,
                            &mut file,
                            &device,
                            override_stop_token_string,
                            config,
                        )?;
                        Ok((model, tokenizer))
                    }
                    Some("ggml" | "bin") | Some(_) | None => {
                        let model = ggml_file::Content::read(&mut file, &device)?;
                        let tokenizer = tokenizer.ok_or(LlamaSourceError::NoTokenizer)?;

                        let gqa = builder.source.group_query_attention;
                        let vocab = tokenizer.get_vocab(true);
                        let start_token_string = match vocab
                            .get("<s>")
                            .map(|v| (*v, "<s>".to_string()))
                            .or_else(|| {
                                vocab
                                    .get("<|start_of_text|>")
                                    .map(|v| (*v, "<|start_of_text|>".to_string()))
                            })
                            .or_else(|| {
                                vocab
                                    .get("<|startoftext|>")
                                    .map(|v| (*v, "<|startoftext|>".to_string()))
                            }) {
                            Some((_, string)) => string,
                            None => String::new(),
                        };
                        let (stop_token, stop_token_string) = match vocab
                            .get("</s>")
                            .map(|v| (*v, "</s>".to_string()))
                            .or_else(|| {
                                vocab
                                    .get("<|end_of_text|>")
                                    .map(|v| (*v, "<|end_of_text|>".to_string()))
                            })
                            .or_else(|| {
                                vocab
                                    .get("<|endoftext|>")
                                    .map(|v| (*v, "<|endoftext|>".to_string()))
                            }) {
                            Some((token, string)) => (token, string),
                            None => return Err(LlamaSourceError::NoStopToken),
                        };
                        let model = Model::from_ggml(
                            model,
                            gqa as usize,
                            &device,
                            start_token_string,
                            stop_token,
                            stop_token_string,
                            config,
                        )?;
                        Ok((model, tokenizer))
                    }
                }
            }
        })
        .await
        .map_err(|_| LlamaSourceError::ModelLoadingPanic)??;

        Ok(Self {
            model,
            tokenizer: Arc::new(tokenizer),
            device,
            merges,
        })
    }

    pub fn _infer(
        &mut self,
        settings: InferenceSettings,
        on_token: Box<dyn FnMut(String) -> Result<(), LlamaModelError> + Send + Sync>,
        finished: &tokio::sync::oneshot::Sender<Result<(), LlamaModelError>>,
    ) -> Result<(), LlamaModelError> {
        let InferenceSettings {
            prompt,
            stop_on,
            sampler,
            mut session,
            max_tokens,
            seed,
        } = settings;

        self._infer_inner(
            prompt,
            stop_on,
            sampler,
            &mut session,
            max_tokens,
            seed,
            on_token,
            finished,
        )
    }

    pub fn _infer_inner(
        &mut self,
        prompt: String,
        stop_on: Option<String>,
        mut sampler: std::sync::Arc<std::sync::Mutex<dyn llm_samplers::prelude::Sampler>>,
        session: &mut LlamaSession,
        max_tokens: u32,
        seed: Option<u64>,
        mut on_token: Box<dyn FnMut(String) -> Result<(), LlamaModelError> + Send + Sync>,
        finished: &tokio::sync::oneshot::Sender<Result<(), LlamaModelError>>,
    ) -> Result<(), LlamaModelError> {
        let mut session = session
            .cache
            .write()
            .map_err(|err| LlamaModelError::Session(err.to_string()))?;

        let tokens = self
            .tokenizer
            .encode_fast(prompt, false)
            .map_err(LlamaModelError::Tokenizer)?;
        let tokens = tokens.get_ids();
        let mut text_stream = TokenOutputStream::new(self.tokenizer.clone());
        for &token in tokens {
            text_stream
                .next_token(token)
                .map_err(LlamaModelError::TokenOutputStreamError)?;
        }

        let mut logit_probs = Vec::new();
        Self::forward(
            &self.model,
            &self.device,
            tokens,
            Some(&mut session),
            &mut logit_probs,
            &self.tokenizer,
        )?;
        let mut logits = Logits::try_from_iter_top_k(logit_probs, 512)
            .expect("model output should be valid logits");
        // This stores a buffer of text that has been generated to check against the stop_on string. It should never be longer than the stop_on string.
        let mut queued_text_matching_stop_on = String::new();
        let stop_on_lowercase = stop_on.as_ref().map(|s| s.to_lowercase());
        let stop_on_lowercase = stop_on_lowercase.as_deref();
        let stop_token = self.model.config.stop_token;
        let mut tokens_generated = 0;
        let mut logit_probs = Vec::new();

        'generate: while !finished.is_closed() && tokens_generated < max_tokens {
            let new_token = text_stream
                .sample_token(&mut sampler, logits, stop_on.as_deref(), seed)
                .map_err(LlamaModelError::TokenOutputStreamError)?;
            Self::forward(
                &self.model,
                &self.device,
                &[new_token],
                Some(&mut session),
                &mut logit_probs,
                &self.tokenizer,
            )?;
            if new_token == stop_token {
                tracing::trace!("Stopping on stop token");
                break;
            }
            if let Some(mut new_text) = text_stream
                .next_token(new_token)
                .map_err(LlamaModelError::TokenOutputStreamError)?
            {
                tokens_generated += 1;
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
                            on_token(before_stop_on)?;
                        }
                        None => {
                            new_text =
                                std::mem::take(&mut queued_text_matching_stop_on) + &new_text;
                            on_token(new_text)?;
                        }
                    }
                } else {
                    on_token(new_text)?;
                }
            }
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

fn normalize_token(token: &str) -> String {
    token
        .replace(" ", "Ġ")
        .replace("\t", "ĉ")
        .replace("\n", "Ċ")
        .replace("\r", "č")
}

fn extract_merges(bytes: &[u8]) -> Result<FxHashSet<[u32; 2]>, LlamaSourceError> {
    #[derive(Debug, Serialize, Deserialize)]
    struct SerializedModel {
        vocab: std::collections::HashMap<String, u32>,
        merges: Vec<String>,
    }
    let json = serde_json::from_slice::<serde_json::Value>(bytes).unwrap();
    let model = json["model"].clone();
    let deserialized = serde_json::from_value::<SerializedModel>(model).unwrap();

    let vocab: HashMap<_, _> = deserialized
        .vocab
        .into_iter()
        .map(|(k, v)| {
            let k = normalize_token(&k);
            (k.as_bytes().to_vec(), v)
        })
        .collect();
    let mut vocab_sorted: Vec<_> = vocab.iter().map(|(k, v)| (k.clone(), *v)).collect();
    vocab_sorted.sort_by_key(|(_, v)| *v);
    let tokens: Vec<_> = vocab_sorted.into_iter().map(|(k, _)| k).collect();

    let merges = deserialized
        .merges
        .into_iter()
        .map(|merge| {
            let (first, second) = merge.split_once(' ').unwrap();
            let first = normalize_token(first);
            let second = normalize_token(second);
            let first_bytes = first.as_bytes();
            let second_bytes = second.as_bytes();
            let merged: Vec<u8> = first_bytes
                .iter()
                .chain(second_bytes.iter())
                .copied()
                .collect();
            let new_token = *vocab.get(&merged).unwrap();
            debug_assert_eq!(merged, tokens[new_token as usize]);
            let first = *vocab.get(first_bytes).unwrap();
            let second = *vocab.get(second_bytes).unwrap();
            [first, second]
        })
        .collect();

    Ok(merges)
}
