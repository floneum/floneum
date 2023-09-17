#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

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

pub struct BertSource {
    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    model_id: String,
    revision: String,
}

impl BertSource {
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            revision: "main".to_string(),
        }
    }

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

pub struct Bert {
    weights_filename: PathBuf,
    tokenizer_filename: PathBuf,
    config: Config,
}

impl Bert {
    pub fn new(source: BertSource) -> anyhow::Result<Self> {
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

        Ok(Self {
            weights_filename,
            tokenizer_filename,
            config,
        })
    }

    pub fn load(&self, options: BertInferenceOptions) -> anyhow::Result<BertInstance> {
        let weights = unsafe { candle_core::safetensors::MmapedFile::new(&self.weights_filename)? };
        let weights = weights.deserialize()?;
        let device = device(options.cpu)?;
        let vb = VarBuilder::from_safetensors(vec![weights], DTYPE, &device);
        let model = BertModel::load(vb, &self.config)?;
        let tokenizer =
            Tokenizer::from_file(&self.tokenizer_filename).map_err(anyhow::Error::msg)?;

        Ok(BertInstance { tokenizer, model })
    }
}

pub struct BertInferenceOptions {
    cpu: bool,
}

impl Default for BertInferenceOptions {
    fn default() -> Self {
        Self { cpu: false }
    }
}

impl BertInferenceOptions {
    pub fn with_cpu(mut self, cpu: bool) -> Self {
        self.cpu = cpu;
        self
    }
}

pub struct BertInstance {
    model: BertModel,
    tokenizer: Tokenizer,
}

impl BertInstance {
    pub fn embed(&mut self, sentences: &[&str]) -> anyhow::Result<Vec<Tensor>> {
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
        println!("running inference on batch {:?}", token_ids.shape());
        let embeddings = self.model.forward(&token_ids, &token_type_ids)?;
        println!("generated embeddings {:?}", embeddings.shape());
        // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = normalize_l2(&embeddings)?;
        println!("pooled embeddings {:?}", embeddings.shape());

        let embeddings = embeddings.chunk(n_sentences, 0)?;

        Ok(embeddings)
    }
}

pub fn normalize_l2(v: &Tensor) -> anyhow::Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

pub fn device(cpu: bool) -> anyhow::Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else {
        let device = Device::cuda_if_available(0)?;
        if !device.is_cuda() {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(device)
    }
}

#[test]
fn embed_sentences() -> anyhow::Result<()> {
    let bert = Bert::new(BertSource::default())?;
    let mut bert = bert.load(BertInferenceOptions::default())?;
    let sentences = vec![
        "Cats are cool",
        "The geopolitical situation is dire",
        "Pets are great",
        "Napoleon was a tyrant",
        "Napoleon was a great general"
    ];
    let embeddings = bert.embed(&sentences)?;
    println!("embeddings {:?}", embeddings);

    // Find the cosine similarity between the first two sentences
    let mut similarities = vec![];
    let n_sentences = sentences.len();
    for (i, e_i) in embeddings.iter().enumerate() {
        for j in (i + 1)..n_sentences {
            let e_j = embeddings.get(j).unwrap();
            let sum_ij = (e_i * e_j)?.sum_all()?.to_scalar::<f32>()?;
            let sum_i2 = (e_i * e_i)?.sum_all()?.to_scalar::<f32>()?;
            let sum_j2 = (e_j * e_j)?.sum_all()?.to_scalar::<f32>()?;
            let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
            similarities.push((cosine_similarity, i, j))
        }
    }
    similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
    for &(score, i, j) in similarities.iter() {
        println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
    }

    Ok(())
}
