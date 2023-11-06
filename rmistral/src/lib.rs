//! # RMistral
//!
//! RMistral is a Rust implementation of the quantized [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/) language model.
//!
//! Mistral 7B is a very small but performant language model that can be easily run on your local machine.
//!
//! This library uses [Candle](https://github.com/huggingface/candle) to run Mistral.
//!
//! ## Usage
//!
//! ```rust
//! use mitral::prelude::*;
//! #[tokio::main]
//! async fn main() {
//!     let mut model = Mistral::default();
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
use llm_samplers::types::Sampler;
use raw::MistralCache;
pub use source::*;

use crate::raw::{Config, Model};
use anyhow::Error as E;
use candle_core::Device;
use hf_hub::{api::sync::Api, Repo, RepoType};
use model::MistralModel;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

/// A prelude of commonly used items in RPhi.
pub mod prelude {
    pub use crate::{Mistral, MistralBuilder, MistralSource};
    pub use kalosm_language_model::*;
}

enum Task {
    Kill,
    Infer {
        settings: InferenceSettings,
        sender: tokio::sync::mpsc::UnboundedSender<String>,
        sampler: Arc<Mutex<dyn Sampler>>,
    },
    RunSync {
        callback: Box<
            dyn for<'a> FnOnce(
                    &'a mut MistralModel,
                )
                    -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>>
                + Send,
        >,
    },
}

/// A quantized Mistral language model with support for streaming generation.
pub struct Mistral {
    task_sender: tokio::sync::mpsc::UnboundedSender<Task>,
    thread_handle: Option<std::thread::JoinHandle<()>>,
    tokenizer: Arc<Tokenizer>,
}

impl Drop for Mistral {
    fn drop(&mut self) {
        self.task_sender.send(Task::Kill).unwrap();
        self.thread_handle.take().unwrap().join().unwrap();
    }
}

impl Default for Mistral {
    fn default() -> Self {
        Mistral::builder().build().unwrap()
    }
}

impl Mistral {
    /// Create a new builder for a Mistral model.
    pub fn builder() -> MistralBuilder {
        MistralBuilder::default()
    }

    /// Check if the model has been downloaded.
    pub(crate) fn downloaded() -> bool {
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn new(model: Model, tokenizer: Tokenizer, device: Device, cache: MistralCache) -> Self {
        let (task_sender, mut task_receiver) = tokio::sync::mpsc::unbounded_channel();
        let arc_tokenizer = Arc::new(tokenizer.clone());

        let thread_handle = std::thread::spawn(move || {
            let mut inner = MistralModel::new(model, tokenizer, device, cache);
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
                                    eprintln!("Error: {}", err);
                                }
                            }
                            Task::RunSync { callback } => {
                                callback(&mut inner).await;
                            }
                        }
                    }
                })
        });
        Self {
            task_sender,
            thread_handle: Some(thread_handle),
            tokenizer: arc_tokenizer,
        }
    }

    /// Get a reference to the tokenizer.
    pub(crate) fn get_tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }

    fn run(
        &mut self,
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

/// A builder with configuration for a Mistral model.
#[derive(Default)]
pub struct MistralBuilder {
    /// Run on CPU rather than on GPU.
    cpu: bool,

    source: source::MistralSource,

    flash_attn: bool,
}

impl MistralBuilder {
    /// Set whether to run on CPU rather than on GPU.
    pub fn with_cpu(mut self, cpu: bool) -> Self {
        self.cpu = cpu;
        self
    }

    /// Set the source for the model.
    pub fn with_source(mut self, source: source::MistralSource) -> Self {
        self.source = source;
        self
    }

    /// Set whether to use Flash Attention.
    pub fn with_flash_attn(mut self, use_flash_attn: bool) -> Self {
        self.flash_attn = use_flash_attn;
        self
    }

    /// Build the model (this will download the model if it is not already downloaded)
    pub fn build(self) -> anyhow::Result<Mistral> {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            self.source.model_id,
            RepoType::Model,
            self.source.revision,
        ));
        let tokenizer_filename = repo.get(&self.source.tokenizer_file)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let device = Device::Cpu;
        let config = Config::config_7b_v0_1(self.flash_attn);
        let filename = repo.get(&self.source.gguf_file)?;
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(filename)?;
        let model = Model::new(&config, vb)?;
        let cache = MistralCache::new(&config);

        Ok(Mistral::new(model, tokenizer, device, cache))
    }
}

#[derive(Debug)]
pub(crate) struct InferenceSettings {
    prompt: String,

    /// The seed to use when generating random samples.
    seed: u64,

    /// The length of the sample to generate (in tokens).
    sample_len: usize,

    /// The token to stop on.
    stop_on: Option<String>,
}

impl InferenceSettings {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            seed: rand::random(),
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
