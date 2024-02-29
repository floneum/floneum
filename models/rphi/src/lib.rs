//! # RPhi
//!
//! RPhi is a Rust implementation of the quantized [Phi 1.5](https://huggingface.co/microsoft/phi-1_5) language model.
//!
//! Phi-1.5 is a very small but performant language model that can be easily run on your local machine.
//!
//! This library uses Quantized Mixformer from [Candle](https://github.com/huggingface/candle) to run Phi-1.5.
//!
//! ## Usage
//!
//! ```rust, no_run
//! use rphi::prelude::*;
//! #[tokio::main]
//! async fn main() {
//!     let mut model = Phi::default();
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

mod language_model;
mod model;
mod raw;
mod source;

use kalosm_common::accelerated_device_if_available;
pub use kalosm_language_model;
use kalosm_language_model::ChatMarkers;
use raw::PhiCache;
pub use source::*;

/// A prelude of commonly used items in RPhi.
pub mod prelude {
    pub use crate::{Phi, PhiBuilder, PhiSource};
    pub use kalosm_language_model::*;
}

use anyhow::Error as E;

use crate::raw::Config;
use crate::raw::MixFormerSequentialForCausalLM as QMixFormer;
use candle_core::Device;
use llm_samplers::prelude::Sampler;
use model::PhiModel;
use std::sync::Arc;
use std::sync::Mutex;
use tokenizers::Tokenizer;

enum Task {
    Kill,
    Infer {
        settings: InferenceSettings,
        sender: tokio::sync::mpsc::UnboundedSender<String>,
        sampler: Arc<Mutex<dyn Sampler>>,
    },
    RunSync {
        callback: SyncCallback,
    },
}

type SyncCallback = Box<
    dyn for<'a> FnOnce(
            &'a mut PhiModel,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>>
        + Send,
>;

/// A quantized Phi-1.5 language model with support for streaming generation.
pub struct Phi {
    task_sender: tokio::sync::mpsc::UnboundedSender<Task>,
    thread_handle: Option<std::thread::JoinHandle<()>>,
    tokenizer: Arc<Tokenizer>,
    chat_markers: Option<ChatMarkers>,
}

impl Drop for Phi {
    fn drop(&mut self) {
        self.task_sender.send(Task::Kill).unwrap();
        self.thread_handle.take().unwrap().join().unwrap();
    }
}

impl Default for Phi {
    fn default() -> Self {
        Phi::builder().build().unwrap()
    }
}

impl Phi {
    /// Create a builder for a Phi model.
    pub fn builder() -> PhiBuilder {
        PhiBuilder::default()
    }

    /// Start the v2 model.
    pub fn v2() -> Self {
        Phi::builder().with_source(PhiSource::v2()).build().unwrap()
    }

    /// Create a new chat model.
    pub fn new_chat() -> Self {
        Phi::builder()
            .with_source(PhiSource::dolphin_phi_v2())
            .build()
            .unwrap()
    }

    /// Check if the model has been downloaded.
    pub(crate) fn downloaded() -> bool {
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        model: QMixFormer,
        tokenizer: Tokenizer,
        device: Device,
        cache: PhiCache,
        chat_markers: Option<ChatMarkers>,
    ) -> Self {
        let (task_sender, mut task_receiver) = tokio::sync::mpsc::unbounded_channel();
        let arc_tokenizer = Arc::new(tokenizer);

        let thread_handle = std::thread::spawn({
            let arc_tokenizer = arc_tokenizer.clone();
            move || {
                let mut inner = PhiModel::new(model, arc_tokenizer, device, cache);
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(async move {
                        while let Some(task) = task_receiver.recv().await {
                            match task {
                                Task::Kill => break,
                                Task::Infer {
                                    settings,
                                    sender,
                                    sampler,
                                } => {
                                    if let Err(err) = inner._infer(settings, sampler, sender) {
                                        tracing::error!("Error in PhiModel::_infer: {}", err);
                                    }
                                }
                                Task::RunSync { callback } => {
                                    callback(&mut inner).await;
                                }
                            }
                        }
                    })
            }
        });
        Self {
            task_sender,
            thread_handle: Some(thread_handle),
            tokenizer: arc_tokenizer,
            chat_markers,
        }
    }

    /// Get the tokenizer used by this model.
    pub(crate) fn get_tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }

    fn run(
        &self,
        settings: InferenceSettings,
        sampler: Arc<Mutex<dyn Sampler>>,
    ) -> anyhow::Result<tokio::sync::mpsc::UnboundedReceiver<String>> {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        self.task_sender
            .send(Task::Infer {
                settings,
                sender,
                sampler,
            })
            .unwrap();
        Ok(receiver)
    }
}

/// A builder with configuration for a Phi model.
#[derive(Default)]
pub struct PhiBuilder {
    /// The source to use for the model.
    source: source::PhiSource,
}

impl PhiBuilder {
    /// Set the source to use for the model.
    pub fn with_source(mut self, source: source::PhiSource) -> Self {
        self.source = source;
        self
    }

    /// Build the model (this will download the model if it is not already downloaded)
    pub fn build(self) -> anyhow::Result<Phi> {
        let tokenizer_filename = self.source.tokenizer.path()?;
        let filename = self.source.model.path()?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let config = self.source.phi_config;
        let device = accelerated_device_if_available()?;
        let vb =
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf(filename, &device)?;
        let model = if self.source.phi2 {
            QMixFormer::new_v2(&config, vb)?
        } else {
            QMixFormer::new(&config, vb)?
        };

        let cache = PhiCache::new(&config);

        Ok(Phi::new(
            model,
            tokenizer,
            device,
            cache,
            self.source.chat_markers,
        ))
    }
}

#[derive(Debug)]
pub(crate) struct InferenceSettings {
    prompt: String,

    /// The length of the sample to generate (in tokens).
    sample_len: usize,

    /// The token to stop on.
    stop_on: Option<String>,
}

impl InferenceSettings {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            sample_len: 100,
            stop_on: None,
        }
    }

    pub fn with_sample_len(mut self, sample_len: usize) -> Self {
        self.sample_len = sample_len;
        self
    }

    pub fn with_stop_on(mut self, stop_on: impl Into<Option<String>>) -> Self {
        self.stop_on = stop_on.into();
        self
    }
}
