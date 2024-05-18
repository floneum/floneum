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
use kalosm_common::*;
pub use language_model::*;

use std::sync::RwLock;

use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::{PaddingParams, Tokenizer};

/// A the source of a [`Bert`] model
pub struct BertSource {
    config: FileSource,
    tokenizer: FileSource,
    model: FileSource,
}

impl BertSource {
    /// Set the model to use, check out available models: <https://huggingface.co/models?library=sentence-transformers&sort=trending>
    pub fn with_model(mut self, model: FileSource) -> Self {
        self.model = model;
        self
    }

    /// Set the tokenizer to use
    pub fn with_tokenizer(mut self, tokenizer: FileSource) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Set the config to use
    pub fn with_config(mut self, config: FileSource) -> Self {
        self.config = config;
        self
    }

    /// Create a new [`BertSource`] with the BGE large english preset
    pub fn bge_large_en() -> Self {
        Self::default()
            .with_model(FileSource::huggingface(
                "BAAI/bge-large-en-v1.5".to_string(),
                "refs/pr/5".to_string(),
                "model.safetensors".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "BAAI/bge-large-en-v1.5".to_string(),
                "refs/pr/5".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "BAAI/bge-large-en-v1.5".to_string(),
                "refs/pr/5".to_string(),
                "config.json".to_string(),
            ))
    }

    /// Create a new [`BertSource`] with the BGE base english preset
    pub fn bge_base_en() -> Self {
        Self::default()
            .with_model(FileSource::huggingface(
                "BAAI/bge-base-en-v1.5".to_string(),
                "refs/pr/1".to_string(),
                "model.safetensors".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "BAAI/bge-base-en-v1.5".to_string(),
                "refs/pr/1".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "BAAI/bge-base-en-v1.5".to_string(),
                "refs/pr/1".to_string(),
                "config.json".to_string(),
            ))
    }

    /// Create a new [`BertSource`] with the BGE small english preset
    pub fn bge_small_en() -> Self {
        Self::default()
            .with_model(FileSource::huggingface(
                "BAAI/bge-small-en-v1.5".to_string(),
                "refs/pr/3".to_string(),
                "model.safetensors".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "BAAI/bge-small-en-v1.5".to_string(),
                "refs/pr/3".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "BAAI/bge-small-en-v1.5".to_string(),
                "refs/pr/3".to_string(),
                "config.json".to_string(),
            ))
    }

    /// Create a new [`BertSource`] with the MiniLM-L6-v2 preset
    pub fn mini_lm_l6_v2() -> Self {
        Self::default()
            .with_model(FileSource::huggingface(
                "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                "refs/pr/21".to_string(),
                "model.safetensors".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                "refs/pr/21".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                "refs/pr/21".to_string(),
                "config.json".to_string(),
            ))
    }

    /// Create a new [`BertSource`] with the [snowflake-arctic-embed-xs](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs) model
    pub fn snowflake_arctic_embed_extra_small() -> Self {
        Self::default()
            .with_model(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-xs".to_string(),
                "main".to_string(),
                "model.safetensors".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-xs".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-xs".to_string(),
                "main".to_string(),
                "config.json".to_string(),
            ))
    }

    /// Create a new [`BertSource`] with the [snowflake-arctic-embed-s](https://huggingface.co/Snowflake/snowflake-arctic-embed-s) model
    pub fn snowflake_arctic_embed_small() -> Self {
        Self::default()
            .with_model(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-s".to_string(),
                "main".to_string(),
                "model.safetensors".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-s".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-s".to_string(),
                "main".to_string(),
                "config.json".to_string(),
            ))
    }

    /// Create a new [`BertSource`] with the [snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) model
    pub fn snowflake_arctic_embed_medium() -> Self {
        Self {
            config: FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-m".to_string(),
                "main".to_string(),
                "config.json".to_string(),
            ),
            tokenizer: FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-m".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            model: FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-m".to_string(),
                "main".to_string(),
                "model.safetensors".to_string(),
            ),
        }
    }

    /// Create a new [`BertSource`] with the [snowflake-arctic-embed-m-long](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long) model
    ///
    /// This model is slightly larger than [`Self::snowflake_arctic_embed_medium`] and supports longer contexts (up to 2048 tokens).
    pub fn snowflake_arctic_embed_medium_long() -> Self {
        Self::default()
            .with_model(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-m-long".to_string(),
                "main".to_string(),
                "model.safetensors".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-m-long".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-m-long".to_string(),
                "main".to_string(),
                "config.json".to_string(),
            ))
    }

    /// Create a new [`BertSource`] with the [snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l) model
    pub fn snowflake_arctic_embed_large() -> Self {
        Self::default()
            .with_model(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-l".to_string(),
                "main".to_string(),
                "model.safetensors".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-l".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "Snowflake/snowflake-arctic-embed-l".to_string(),
                "main".to_string(),
                "config.json".to_string(),
            ))
    }
}

impl Default for BertSource {
    fn default() -> Self {
        Self::snowflake_arctic_embed_medium()
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
    pub async fn build(self) -> anyhow::Result<Bert> {
        self.build_with_loading_handler(ModelLoadingProgress::multi_bar_loading_indicator())
            .await
    }

    /// Build the model with a loading handler
    pub async fn build_with_loading_handler(
        self,
        loading_handler: impl FnMut(ModelLoadingProgress) + Send + 'static,
    ) -> anyhow::Result<Bert> {
        Bert::from_builder(self, loading_handler).await
    }
}

/// A bert model
pub struct Bert {
    model: BertModel,
    tokenizer: RwLock<Tokenizer>,
}

impl Bert {
    /// Create a new [`BertBuilder`]
    pub fn builder() -> BertBuilder {
        BertBuilder::default()
    }

    /// Create a new default bert model
    pub async fn new() -> anyhow::Result<Self> {
        Self::builder().build().await
    }

    async fn from_builder(
        builder: BertBuilder,
        mut progress_handler: impl FnMut(ModelLoadingProgress) + Send + 'static,
    ) -> anyhow::Result<Self> {
        let BertBuilder { source } = builder;
        let BertSource {
            config,
            tokenizer,
            model,
        } = source;

        let source = format!("Config ({})", config);
        let mut create_progress = ModelLoadingProgress::downloading_progress(source);
        let config_filename = config
            .download(|progress| progress_handler(create_progress(progress)))
            .await?;
        let tokenizer_source = format!("Tokenizer ({})", tokenizer);
        let mut create_progress = ModelLoadingProgress::downloading_progress(tokenizer_source);
        let tokenizer_filename = tokenizer
            .download(|progress| progress_handler(create_progress(progress)))
            .await?;
        let model_source = format!("Model ({})", model);
        let mut create_progress = ModelLoadingProgress::downloading_progress(model_source);
        let weights_filename = model
            .download(|progress| progress_handler(create_progress(progress)))
            .await?;

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
