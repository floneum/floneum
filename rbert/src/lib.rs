//! # rbert
//!
//! A Rust wrapper for [bert sentence transformers](https://arxiv.org/abs/1908.10084) implemented in [Candle](https://github.com/huggingface/candle)
//!
//! ## Usage
//!
//! ```rust
//! use floneumin_language_model::Embedder;
//! use rbert::*;
//! 
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut bert = Bert::builder().build()?;
//!     let sentences = vec![
//!         "Cats are cool",
//!         "The geopolitical situation is dire",
//!         "Pets are great",
//!         "Napoleon was a tyrant",
//!         "Napoleon was a great general",
//!     ];
//!     let embeddings = bert.embed_batch(&sentences).await?;
//!     println!("embeddings {:?}", embeddings);
//! 
//!     // Find the cosine similarity between the first two sentences
//!     let mut similarities = vec![];
//!     let n_sentences = sentences.len();
//!     for (i, e_i) in embeddings.iter().enumerate() {
//!         for j in (i + 1)..n_sentences {
//!             let e_j = embeddings.get(j).unwrap();
//!             let cosine_similarity = e_j.cosine_similarity(e_i);
//!             similarities.push((cosine_similarity, i, j))
//!         }
//!     }
//!     similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
//!     for &(score, i, j) in similarities.iter() {
//!         println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
//!     }
//! 
//!     Ok(())
//! }
//! ```


#![warn(missing_docs)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod language_model;
pub use language_model::*;

use std::path::PathBuf;

use anyhow::anyhow;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Cache, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

fn try_fetch(repo: Repo) -> anyhow::Result<(PathBuf, PathBuf, PathBuf)> {
    let api = Api::new()?;
    let api = api.repo(repo);
    Ok((
        api.get("config.json")?,
        api.get("tokenizer.json")?,
        api.get("model.safetensors")?,
    ))
}

/// A the source of a [`Bert`] model
pub struct BertSource {
    model_id: String,
    revision: String,
}

impl BertSource {
    /// Set the model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    pub fn with_model_id(mut self, model_id: String) -> Self {
        self.model_id = model_id;
        self
    }

    /// Set the revision to use
    pub fn with_revision(mut self, revision: String) -> Self {
        self.revision = revision;
        self
    }
}

impl Default for BertSource {
    fn default() -> Self {
        let default_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let default_revision = "refs/pr/21".to_string();
        Self {
            model_id: default_model,
            revision: default_revision,
        }
    }
}

/// A builder for a [`Bert`] model
pub struct BertBuilder {
    source: BertSource,
    cpu: bool,
}

impl Default for BertBuilder {
    fn default() -> Self {
        Self {
            source: Default::default(),
            cpu: false,
        }
    }
}

impl BertBuilder {
    /// Set the source of the model
    pub fn with_source(mut self, source: BertSource) -> Self {
        self.source = source;
        self
    }

    /// Set whether to use the CPU or GPU
    pub fn with_cpu(mut self, cpu: bool) -> Self {
        self.cpu = cpu;
        self
    }

    /// Build the model
    pub fn build(self) -> anyhow::Result<Bert> {
        Bert::new(self)
    }
}

/// A bert model
pub struct Bert {
    model: BertModel,
    tokenizer: Tokenizer,
}

impl Default for Bert {
    fn default() -> Self {
        Self::builder().build().unwrap()
    }
}

impl Bert {
    /// Create a new [`BertBuilder`]
    pub fn builder() -> BertBuilder {
        BertBuilder::default()
    }

    fn new(builder: BertBuilder) -> anyhow::Result<Self> {
        let BertBuilder { source, cpu } = builder;
        let BertSource { model_id, revision } = source;

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = match try_fetch(repo.clone())
        {
            Ok(filenames) => filenames,
            Err(err) => {
                tracing::warn!(
                    "Failed to fetch model from hub, falling back to local cache: {}",
                    err
                );
                let cache = Cache::default().repo(repo);
                (
                    cache
                        .get("config.json")
                        .ok_or(anyhow!("Missing config file in cache"))?,
                    cache
                        .get("tokenizer.json")
                        .ok_or(anyhow!("Missing tokenizer file in cache"))?,
                    cache
                        .get("model.safetensors")
                        .ok_or(anyhow!("Missing weights file in cache"))?,
                )
            }
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;

        let device = device(cpu)?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&weights_filename], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_filename).map_err(anyhow::Error::msg)?;

        Ok(Bert { tokenizer, model })
    }

    /// Embed a batch of sentences
    pub(crate) fn embed_batch_raw(&mut self, sentences: &[&str]) -> anyhow::Result<Vec<Tensor>> {
        let mut combined = Vec::new();
        for batch in sentences.chunks(4) {
            let embeddings = self.embed_batch_raw_inner(batch)?;
            combined.extend(embeddings);
        }
        Ok(combined)
    }

    fn embed_batch_raw_inner(&mut self, sentences: &[&str]) -> anyhow::Result<Vec<Tensor>> {
        let device = &self.model.device;

        let n_sentences = sentences.len();
        if let Some(pp) = self.tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            self.tokenizer.with_padding(Some(pp));
        }
        let tokens = self
            .tokenizer
            .encode_batch(sentences.to_vec(), true)
            .map_err(anyhow::Error::msg)?;
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), device)?)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let embeddings = self.model.forward(&token_ids, &token_type_ids)?;
        // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = normalize_l2(&embeddings)?;
        let embeddings = embeddings.chunk(n_sentences, 0)?;

        Ok(embeddings)
    }
}

fn normalize_l2(v: &Tensor) -> anyhow::Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

fn device(cpu: bool) -> anyhow::Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else {
        let device = Device::cuda_if_available(0)?;
        if !device.is_cuda() {
            tracing::warn!(
                "Running on CPU, to run on GPU, build this example with `--features cuda`"
            );
        }
        Ok(device)
    }
}
