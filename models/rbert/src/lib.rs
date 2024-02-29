//! # rbert
//!
//! A Rust wrapper for [bert sentence transformers](https://arxiv.org/abs/1908.10084) implemented in [Candle](https://github.com/huggingface/candle)
//!
//! ## Usage
//!
//! ```rust, no_run
//! use kalosm_language_model::Embedder;
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

use kalosm_common::accelerated_device_if_available;
use std::{path::PathBuf, sync::RwLock};

use anyhow::anyhow;
use candle_core::Tensor;
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
    /// Set the model to use, check out available models: <https://huggingface.co/models?library=sentence-transformers&sort=trending>
    pub fn with_model_id(mut self, model_id: String) -> Self {
        self.model_id = model_id;
        self
    }

    /// Set the revision to use
    pub fn with_revision(mut self, revision: String) -> Self {
        self.revision = revision;
        self
    }

    /// Create a new [`BertSource`] with the BGE large english preset
    pub fn bge_large_en() -> Self {
        Self::default()
            .with_model_id("BAAI/bge-large-en-v1.5".to_string())
            .with_revision("refs/pr/5".to_string())
    }

    /// Create a new [`BertSource`] with the BGE base english preset
    pub fn bge_base_en() -> Self {
        Self::default()
            .with_model_id("BAAI/bge-base-en-v1.5".to_string())
            .with_revision("refs/pr/1".to_string())
    }

    /// Create a new [`BertSource`] with the BGE small english preset
    pub fn bge_small_en() -> Self {
        Self::default()
            .with_model_id("BAAI/bge-small-en-v1.5".to_string())
            .with_revision("refs/pr/3".to_string())
    }

    /// Create a new [`BertSource`] with the MiniLM-L6-v2 preset
    pub fn mini_lm_l6_v2() -> Self {
        Self::default()
            .with_model_id("sentence-transformers/all-MiniLM-L6-v2".to_string())
            .with_revision("refs/pr/21".to_string())
    }
}

impl Default for BertSource {
    fn default() -> Self {
        Self {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            revision: "refs/pr/21".to_string(),
        }
    }
}

/// A builder for a [`Bert`] model
#[derive(Default)]
pub struct BertBuilder {
    source: BertSource,
}

impl BertBuilder {
    /// Set the source of the model
    pub fn with_source(mut self, source: BertSource) -> Self {
        self.source = source;
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
    tokenizer: RwLock<Tokenizer>,
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
        let BertBuilder { source } = builder;
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

        let device = accelerated_device_if_available()?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&weights_filename], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_filename).map_err(anyhow::Error::msg)?;

        Ok(Bert {
            tokenizer: RwLock::new(tokenizer),
            model,
        })
    }

    /// Embed a batch of sentences
    pub(crate) fn embed_batch_raw(&self, sentences: &[&str]) -> anyhow::Result<Vec<Tensor>> {
        let mut combined = Vec::new();
        for batch in sentences.chunks(4) {
            let embeddings = self.embed_batch_raw_inner(batch)?;
            combined.extend(embeddings);
        }
        Ok(combined)
    }

    fn embed_batch_raw_inner(&self, sentences: &[&str]) -> anyhow::Result<Vec<Tensor>> {
        let device = &self.model.device;

        let n_sentences = sentences.len();
        let tokens = {
            let mut tokenizer_write = self.tokenizer.write().unwrap();
            if let Some(pp) = tokenizer_write.get_padding_mut() {
                pp.strategy = tokenizers::PaddingStrategy::BatchLongest
            } else {
                let pp = PaddingParams {
                    strategy: tokenizers::PaddingStrategy::BatchLongest,
                    ..Default::default()
                };
                tokenizer_write.with_padding(Some(pp));
            }
            tokenizer_write
                .encode_batch(sentences.to_vec(), true)
                .map_err(anyhow::Error::msg)?
        };
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
