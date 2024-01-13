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
//! ```rust, ignore // TODO: Restore
//! use kalosm_llama::prelude::*;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut model = Llama::default();
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

// mod language_model;
// mod model;
mod raw;
// mod session;
// mod source;

// pub use crate::model::LlamaModel;
// use crate::raw::Model;
// pub use crate::session::LlamaSession;
// use candle_core::{
//     quantized::{ggml_file, gguf_file},
//     Device,
// };
// use kalosm_language_model::ChatMarkers;
// use llm_samplers::types::Sampler;
// use session::LlamaCache;
// pub use source::*;
// use std::sync::{Arc, Mutex};
// use tokenizers::Tokenizer;

// /// A prelude of commonly used items in RPhi.
// pub mod prelude {
//     pub use crate::session::LlamaSession;
//     pub use crate::{Llama, LlamaBuilder, LlamaSource};
//     pub use kalosm_language_model::*;
// }

// enum Task {
//     Kill,
//     Infer {
//         settings: InferenceSettings,
//         sender: tokio::sync::mpsc::UnboundedSender<String>,
//         sampler: Arc<Mutex<dyn Sampler>>,
//     },
//     RunSync {
//         callback: SyncCallback,
//     },
// }

// type SyncCallback = Box<
//     dyn for<'a> FnOnce(
//             &'a mut LlamaModel,
//         ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>>
//         + Send,
// >;

// /// A quantized Llama language model with support for streaming generation.
// pub struct Llama {
//     task_sender: tokio::sync::mpsc::UnboundedSender<Task>,
//     thread_handle: Option<std::thread::JoinHandle<()>>,
//     tokenizer: Arc<Tokenizer>,
//     chat_markers: Option<ChatMarkers>,
// }

// impl Drop for Llama {
//     fn drop(&mut self) {
//         self.task_sender.send(Task::Kill).unwrap();
//         self.thread_handle.take().unwrap().join().unwrap();
//     }
// }

// impl Default for Llama {
//     fn default() -> Self {
//         Llama::builder().build().unwrap()
//     }
// }

// impl Llama {
//     /// Create a default chat model.
//     pub fn new_chat() -> Self {
//         Llama::builder()
//             .with_source(LlamaSource::open_chat_7b())
//             .build()
//             .unwrap()
//     }

//     /// Create a new builder for a Llama model.
//     pub fn builder() -> LlamaBuilder {
//         LlamaBuilder::default()
//     }

//     /// Check if the model has been downloaded.
//     pub(crate) fn downloaded() -> bool {
//         false
//     }

//     #[allow(clippy::too_many_arguments)]
//     fn new(
//         model: Model,
//         tokenizer: Tokenizer,
//         device: Device,
//         cache: LlamaCache,
//         chat_markers: Option<ChatMarkers>,
//     ) -> Self {
//         let (task_sender, mut task_receiver) = tokio::sync::mpsc::unbounded_channel();
//         let arc_tokenizer = Arc::new(tokenizer);

//         let thread_handle = std::thread::spawn({
//             let arc_tokenizer = arc_tokenizer.clone();
//             move || {
//                 let mut inner = LlamaModel::new(model, arc_tokenizer, device, cache);
//                 tokio::runtime::Builder::new_current_thread()
//                     .enable_all()
//                     .build()
//                     .unwrap()
//                     .block_on(async move {
//                         while let Some(task) = task_receiver.recv().await {
//                             match task {
//                                 Task::Kill => break,
//                                 Task::Infer {
//                                     settings,
//                                     sender,
//                                     sampler,
//                                 } => {
//                                     if let Err(err) = inner._infer(settings, sampler, sender) {
//                                         eprintln!("Error: {}", err);
//                                     }
//                                 }
//                                 Task::RunSync { callback } => {
//                                     callback(&mut inner).await;
//                                 }
//                             }
//                         }
//                     })
//             }
//         });
//         Self {
//             task_sender,
//             thread_handle: Some(thread_handle),
//             tokenizer: arc_tokenizer,
//             chat_markers,
//         }
//     }

//     /// Get a reference to the tokenizer.
//     pub(crate) fn get_tokenizer(&self) -> Arc<Tokenizer> {
//         self.tokenizer.clone()
//     }

//     fn run(
//         &self,
//         settings: InferenceSettings,
//         sampler: Arc<Mutex<dyn Sampler>>,
//     ) -> anyhow::Result<tokio::sync::mpsc::UnboundedReceiver<String>> {
//         let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
//         self.task_sender
//             .send(Task::Infer {
//                 settings,
//                 sender,
//                 sampler,
//             })
//             .unwrap();
//         Ok(receiver)
//     }
// }

// /// A builder with configuration for a Llama model.
// #[derive(Default)]
// pub struct LlamaBuilder {
//     /// Run on CPU rather than on GPU.
//     cpu: bool,

//     source: source::LlamaSource,

//     flash_attn: bool,
// }

// impl LlamaBuilder {
//     /// Set whether to run on CPU rather than on GPU.
//     pub fn with_cpu(mut self, cpu: bool) -> Self {
//         self.cpu = cpu;
//         self
//     }

//     /// Set the source for the model.
//     pub fn with_source(mut self, source: source::LlamaSource) -> Self {
//         self.source = source;
//         self
//     }

//     /// Set whether to use Flash Attention.
//     pub fn with_flash_attn(mut self, use_flash_attn: bool) -> Self {
//         self.flash_attn = use_flash_attn;
//         self
//     }

//     /// Build the model (this will download the model if it is not already downloaded)
//     pub fn build(self) -> anyhow::Result<Llama> {
//         let tokenizer = self.source.tokenizer()?;

//         let device = Device::cuda_if_available(0)?;
//         let filename = self.source.model()?;
//         let mut file = std::fs::File::open(&filename)?;
//         let model = match filename.extension().and_then(|v| v.to_str()) {
//             Some("gguf") => {
//                 let model = gguf_file::Content::read(&mut file)?;
//                 Model::from_gguf(model, &mut file)?
//             }
//             Some("ggml" | "bin") | Some(_) | None => {
//                 let model = ggml_file::Content::read(&mut file)?;
//                 let gqa = self.source.group_query_attention;
//                 Model::from_ggml(model, gqa as usize)?
//             }
//         };

//         let cache = LlamaCache::new(&model);

//         Ok(Llama::new(
//             model,
//             tokenizer,
//             device,
//             cache,
//             self.source.markers,
//         ))
//     }
// }

// #[derive(Debug)]
// pub(crate) struct InferenceSettings {
//     prompt: String,

//     /// The length of the sample to generate (in tokens).
//     sample_len: usize,

//     /// The token to stop on.
//     stop_on: Option<String>,
// }

// impl InferenceSettings {
//     pub fn new(prompt: impl Into<String>) -> Self {
//         Self {
//             prompt: prompt.into(),
//             sample_len: 100,
//             stop_on: None,
//         }
//     }

//     pub fn with_sample_len(mut self, sample_len: usize) -> Self {
//         self.sample_len = sample_len;
//         self
//     }

//     pub fn with_stop_on(mut self, stop_on: impl Into<Option<String>>) -> Self {
//         self.stop_on = stop_on.into();
//         self
//     }
// }
