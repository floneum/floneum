//! # RLlama
//!
//! RLlama is a Rust implementation of the quantized [Llama 7B](https://llama.ai/news/announcing-llama-7b/) language model.
//!
//! Llama 7B is a very small but performant language model that can be easily run on your local machine.
//!
//! This library uses [Candle](https://github.com/huggingface/candle) to run Llama.
//!
//! ## Usage
//!
//! ```rust, no_run
//! use kalosm_llama::prelude::*;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut model = Llama::new().await.unwrap();
//!     let prompt = "The capital of France is ";
//!     let mut result = model.stream_text(prompt).await.unwrap();
//!
//!     print!("{prompt}");
//!     while let Some(token) = result.next().await {
//!         print!("{token}");
//!     }
//! }
//! ```

#![warn(missing_docs)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

// mod chat_template;
mod language_model;
mod model;
mod raw;
mod session;
mod source;
// mod structured;
mod token_stream;

pub use crate::model::LlamaModel;
pub use crate::raw::cache::*;
use crate::raw::Model;
pub use crate::session::LlamaSession;
use candle_core::{
    quantized::{ggml_file, gguf_file},
    Device,
};
pub use kalosm_common::*;
use llm_samplers::types::Sampler;
use model::LlamaModelError;
use raw::LlamaConfig;
pub use source::*;
use std::{
    any::Any,
    sync::{Arc, Mutex},
};
use tokenizers::Tokenizer;

/// A prelude of commonly used items in kalosm-llama.
pub mod prelude {
    pub use crate::session::LlamaSession;
    pub use crate::{Llama, LlamaBuilder, LlamaSource};
    pub use kalosm_language_model::*;
}

struct Task {
    settings: InferenceSettings,
    on_token: Box<dyn FnMut(String) -> Result<(), LlamaModelError> + Send + Sync>,
    finished: tokio::sync::oneshot::Sender<Result<Box<dyn Any + Send>, LlamaModelError>>,
}

type SyncCallback = Box<
    dyn for<'a> FnOnce(
            &'a mut LlamaModel,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>>
        + Send,
>;

/// A quantized Llama language model with support for streaming generation.
#[derive(Clone)]
pub struct Llama {
    config: Arc<LlamaConfig>,
    task_sender: tokio::sync::mpsc::UnboundedSender<Task>,
}

impl Llama {
    /// Create a default chat model.
    pub async fn new_chat() -> Result<Self, LlamaSourceError> {
        Llama::builder()
            .with_source(LlamaSource::llama_3_1_8b_chat())
            .build()
            .await
    }

    /// Create a default phi-3 chat model.
    pub async fn phi_3() -> Result<Self, LlamaSourceError> {
        Llama::builder()
            .with_source(LlamaSource::phi_3_5_mini_4k_instruct())
            .build()
            .await
    }

    /// Create a default text generation model.
    pub async fn new() -> Result<Self, LlamaSourceError> {
        Llama::builder()
            .with_source(LlamaSource::llama_8b())
            .build()
            .await
    }

    /// Create a new builder for a Llama model.
    pub fn builder() -> LlamaBuilder {
        LlamaBuilder::default()
    }

    #[allow(clippy::too_many_arguments)]
    fn from_build(mut model: LlamaModel) -> Self {
        let (task_sender, mut task_receiver) = tokio::sync::mpsc::unbounded_channel();
        let config = model.model.config.clone();

        std::thread::spawn({
            move || {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(async move {
                        while let Some(Task {
                            settings,
                            on_token,
                            finished,
                        }) = task_receiver.recv().await
                        {
                            if let Err(err) = model._infer(settings, on_token, finished) {
                                tracing::error!("Error running Llama model: {}", err);
                            }
                        }
                    })
            }
        });
        Self {
            task_sender,
            config,
        }
    }

    fn run(
        &self,
        settings: InferenceSettings,
        on_token: Box<dyn FnMut(String) -> Result<(), LlamaModelError> + Send + Sync>,
        finished: tokio::sync::oneshot::Sender<Result<Box<dyn Any + Send>, LlamaModelError>>,
    ) -> Result<tokio::sync::mpsc::UnboundedReceiver<String>, LlamaModelError> {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        _ = self.task_sender.send(Task {
            settings,
            on_token,
            finished,
        });
        Ok(receiver)
    }
}

/// A builder with configuration for a Llama model.
#[derive(Default)]
pub struct LlamaBuilder {
    source: source::LlamaSource,
    device: Option<Device>,
    flash_attn: bool,
}

impl LlamaBuilder {
    /// Set the source for the model.
    pub fn with_source(mut self, source: source::LlamaSource) -> Self {
        self.source = source;
        self
    }

    /// Set whether to use Flash Attention.
    pub fn with_flash_attn(mut self, use_flash_attn: bool) -> Self {
        self.flash_attn = use_flash_attn;
        self
    }

    /// Set the device to run the model with. (Defaults to an accelerator if available, otherwise the CPU)
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Get the device or the default device if not set.
    pub(crate) fn get_device(&self) -> Result<Device, LlamaSourceError> {
        match self.device.clone() {
            Some(device) => Ok(device),
            None => Ok(accelerated_device_if_available()?),
        }
    }

    /// Build the model with a handler for progress as the download and loading progresses.
    ///
    /// ```rust, no_run
    /// use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), anyhow::Error> {
    /// // Create a new llama model with a loading handler
    /// let model = Llama::builder()
    ///     .build_with_loading_handler(|progress| match progress {
    ///         ModelLoadingProgress::Downloading {
    ///             source,
    ///             start_time,
    ///             progress,
    ///         } => {
    ///             let progress = (progress * 100.0) as u32;
    ///             let elapsed = start_time.elapsed().as_secs_f32();
    ///             println!("Downloading file {source} {progress}% ({elapsed}s)");
    ///         }
    ///         ModelLoadingProgress::Loading { progress } => {
    ///             let progress = (progress * 100.0) as u32;
    ///             println!("Loading model {progress}%");
    ///         }
    ///     })
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn build_with_loading_handler(
        self,
        handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Llama, LlamaSourceError> {
        let model = LlamaModel::from_builder(self, handler).await?;

        Ok(Llama::from_build(model))
    }

    /// Build the model (this will download the model if it is not already downloaded)
    pub async fn build(self) -> Result<Llama, LlamaSourceError> {
        self.build_with_loading_handler(ModelLoadingProgress::multi_bar_loading_indicator())
            .await
    }
}

#[derive(Debug)]
pub(crate) struct InferenceSettings {
    prompt: String,

    /// The token to stop on.
    stop_on: Option<String>,

    /// The sampler to use.
    sampler: Option<std::sync::Arc<std::sync::Mutex<dyn llm_samplers::prelude::Sampler>>>,

    /// The session to use.
    session: LlamaSession,
}

impl InferenceSettings {
    pub fn new(prompt: impl Into<String>, session: LlamaSession) -> Self {
        Self {
            prompt: prompt.into(),
            stop_on: None,
            sampler: None,
            session,
        }
    }

    pub fn with_stop_on(mut self, stop_on: impl Into<Option<String>>) -> Self {
        self.stop_on = stop_on.into();
        self
    }

    pub fn with_sampler(mut self, sampler: impl Sampler + 'static) -> Self {
        self.sampler = Some(std::sync::Arc::new(std::sync::Mutex::new(sampler)));
        self
    }
}
