use std::sync::Arc;

use crate::chat_template::HuggingFaceChatTemplate;
use crate::raw::attention_layer::LlamaAttention;
use crate::raw::rope::RopeCache;
use crate::LlamaSourceError;
use attention_layer::AttentionBias;
use attention_layer::AttentionVariant;
use attention_layer::FeedForwardVariant;
use attention_layer::GroupedAttention;
use attention_layer::LlamaFeedForward;
use attention_layer::PhiFeedForward;
use attention_layer::SeparateAttention;
use fusor_core::QMatrix;
use fusor_core::Sum;
use fusor_core::{Device, Result, Tensor};
use fusor_gguf::GgufMetadata;
use fusor_gguf::GgufValue;

mod attention_layer;
pub mod cache;
mod mask;
mod rope;

use cache::LlamaCache;
use mask::MaskCache;

pub struct Embedding {
    weights: Tensor<2, f32>,
    hidden_size: usize,
}

impl Embedding {
    pub fn new(weights: QMatrix, hidden_size: usize) -> Self {
        let weights = weights.dequantize();
        Self {
            weights,
            hidden_size,
        }
    }

    pub fn forward(&self, indexes: &Tensor<1, u32>) -> Tensor<2, f32> {
        let final_dims = [indexes.shape()[0], self.hidden_size];
        let values = self.weights.index_select(0, indexes);
        values.reshape(final_dims)
    }
}

struct RmsNorm {
    weights: QMatrix,
    eps: f32,
}

impl RmsNorm {
    fn from_qtensor(weights: QMatrix, eps: f32) -> Result<Self> {
        Ok(Self { weights, eps })
    }

    fn forward(&self, x: &Tensor<2, f32>) -> Tensor<2, f32> {
        let shape = *x.shape();
        // Create a sum of everything but the last dimension
        let last_dim_size = shape[1] as f32;
        debug_assert!(last_dim_size > 0.);
        let norm = x.sqr().sum(1) / last_dim_size;
        // Divide the input tensor by the sqrt of the sum plus the epsilon
        let x = x.clone() / (norm + self.eps).sqrt().broadcast(shape);
        // Finally, multiply the result by the weights
        x * self.weights.dequantize::<1, _>().broadcast(shape)
    }
}

/// The configuration of a Llama model.
pub struct LlamaConfig {
    rope_freq_weight: Option<Tensor<2, f32>>,
    rope_theta: f32,
    pub(crate) context_length: usize,
    head_dimension: usize,
    n_head: usize,
    pub(crate) n_layer: usize,
    pub(crate) start_token_string: String,
    pub(crate) stop_token: u32,
    pub(crate) stop_token_string: String,
    pub(crate) chat_template: Option<HuggingFaceChatTemplate>,
    pub(crate) rope_scaling: Option<RopeScalingConfig>,
}

impl LlamaConfig {
    fn hidden_size(&self) -> usize {
        self.head_dimension * self.n_head
    }

    #[cfg(test)]
    pub(crate) fn mock_test() -> Self {
        Self {
            rope_freq_weight: None,
            rope_theta: 5000.,
            context_length: 6,
            head_dimension: 2,
            n_head: 0,
            n_layer: 0,
            start_token_string: "<|startoftext|>".to_string(),
            stop_token: 0,
            stop_token_string: "<|endoftext|>".to_string(),
            chat_template: None,
            rope_scaling: None,
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeScalingConfig {
    pub(crate) factor: f32,
    pub(crate) high_freq_factor: f32,
    pub(crate) low_freq_factor: f32,
    pub(crate) original_max_position_embeddings: usize,
}

pub struct Model {
    pub(crate) config: Arc<LlamaConfig>,
    tok_embeddings: Embedding,
    layers: Vec<LlamaAttention>,
    norm: RmsNorm,
    output: QMatrix,
    masks: MaskCache,
}

impl Model {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &GgufMetadata,
        reader: &mut R,
        device: &Device,
        override_stop_token_string: Option<String>,
        rope_scaling: Option<RopeScalingConfig>,
    ) -> std::result::Result<Self, LlamaSourceError> {
        let md_get = |s: &str| {
            let value = if s.starts_with('.') {
                ct.metadata
                    .iter()
                    .find_map(|(k, value)| k.ends_with(s).then_some(value))
            } else {
                ct.metadata.get(s)
            };
            match value {
                None => Err(LlamaSourceError::MissingGgufEntry(s.to_string())),
                Some(v) => Ok(v),
            }
        };

        let md_tensor = |s: &str, reader: &mut R| {
            QMatrix::read_from_file(device, ct, reader, s)?
                .ok_or_else(|| LlamaSourceError::MissingGgufEntry(s.to_string()))
        };

        // Get the eos and bos tokens from the metadata
        let tokens: Box<[GgufValue]> = md_get("tokenizer.ggml.tokens")?.clone().try_into()?;
        let tokens: Result<Vec<Box<str>>, LlamaSourceError> = tokens
            .iter()
            .map(|v| {
                let v: Box<str> = v.try_into()?;
                Ok(v)
            })
            .collect();
        let tokens = tokens?;
        let start_token: Option<u32> = md_get("tokenizer.ggml.bos_token_id")
            .ok()
            .and_then(|v| v.try_into().ok());
        let stop_token = if let Some(override_stop_token_string) = override_stop_token_string {
            tokens
                .iter()
                .position(|v| &**v == override_stop_token_string)
                .unwrap_or(0) as u32
        } else {
            md_get("tokenizer.ggml.eos_token_id")?.clone().try_into()?
        };
        let start_token_string = start_token
            .map(|v| tokens[v as usize].to_string())
            .unwrap_or_else(|| "".to_string());
        let stop_token_string = tokens[stop_token as usize].to_string();
        let chat_template: Option<Box<str>> = md_get("tokenizer.chat_template")
            .ok()
            .and_then(|v| v.try_into().ok());
        let chat_template = match chat_template {
            Some(chat_template) => {
                let chat_template = HuggingFaceChatTemplate::create(chat_template)
                    .map_err(LlamaSourceError::ChatTemplate)?;
                Some(chat_template)
            }
            None => None,
        };

        // Parameter extraction from metadata.
        let head_count: u32 = md_get(".attention.head_count")?.try_into()?;
        let head_count = head_count as usize;
        let head_count_kv: u32 = md_get(".attention.head_count_kv")?.try_into()?;
        let head_count_kv = head_count_kv as usize;
        let block_count: u32 = md_get(".block_count")?.try_into()?;
        let block_count = block_count as usize;
        let embedding_length: u32 = md_get(".embedding_length")?.try_into()?;
        let embedding_length = embedding_length as usize;
        // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
        let rms_norm_eps: f64 = md_get(".attention.layer_norm_rms_epsilon")?.try_into()?;

        let rope_freq_base = md_get(".rope.freq_base")
            .and_then(|m| Ok(m.clone().try_into()?))
            .unwrap_or(10_000f32);

        let context_length: u32 = md_get(".context_length")?.try_into()?;
        let context_length = context_length as usize;
        let head_dim = embedding_length / head_count;

        let config = LlamaConfig {
            rope_freq_weight: QMatrix::read_from_file(device, ct, reader, "rope_freqs.weight")?
                .map(|q| q.dequantize()),
            rope_theta: rope_freq_base,
            context_length,
            head_dimension: head_dim,
            n_head: head_count,
            n_layer: block_count,
            start_token_string,
            stop_token,
            stop_token_string,
            chat_template,
            rope_scaling,
        };
        let config = Arc::new(config);

        let rope = RopeCache::new(&config, device);

        let tok_embeddings_q = md_tensor("token_embd.weight", reader)?;

        let norm = md_tensor("output_norm.weight", reader)?;
        let norm = RmsNorm::from_qtensor(norm, rms_norm_eps as f32)?;
        let output =
            QMatrix::read_from_file(device, ct, reader, "output.weight")?.unwrap_or_else(|| {
                // If there is no output layer, assume the word embeddings are tied to the output
                tok_embeddings_q.clone()
            });
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_variant = if let Some(qkv) =
                QMatrix::read_from_file(device, ct, reader, &format!("{prefix}.attn_qkv.weight"))?
            {
                AttentionVariant::Grouped(GroupedAttention { attention_qkv: qkv })
            } else {
                let q = md_tensor(&format!("{prefix}.attn_q.weight"), reader)?;
                let k = md_tensor(&format!("{prefix}.attn_k.weight"), reader)?;
                let v = md_tensor(&format!("{prefix}.attn_v.weight"), reader)?;
                let bias = if let (Some(bias_q), Some(bias_k), Some(bias_v)) = (
                    QMatrix::read_from_file(device, ct, reader, &format!("{prefix}.attn_q.bias"))?,
                    QMatrix::read_from_file(device, ct, reader, &format!("{prefix}.attn_k.bias"))?,
                    QMatrix::read_from_file(device, ct, reader, &format!("{prefix}.attn_v.bias"))?,
                ) {
                    Some(AttentionBias {
                        bias_q: bias_q.dequantize(),
                        bias_k: bias_k.dequantize(),
                        bias_v: bias_v.dequantize(),
                    })
                } else {
                    None
                };
                let architecture: Box<str> = md_get("general.architecture")?.try_into()?;
                let separate = SeparateAttention {
                    attention_wq: q,
                    attention_wk: k,
                    attention_wv: v,
                    interleaved_rope: architecture.as_ref() != "qwen2",
                    bias,
                };
                AttentionVariant::Separate(separate)
            };
            let attention_wo = md_tensor(&format!("{prefix}.attn_output.weight"), reader)?;
            // Try to read from the up, down and gate weights
            let feed_forward_variant = if let Some(ffn_gate) =
                QMatrix::read_from_file(device, ct, reader, &format!("{prefix}.ffn_gate.weight"))?
            {
                let feed_forward_w1 = ffn_gate;
                let feed_forward_w2 = md_tensor(&format!("{prefix}.ffn_down.weight"), reader)?;
                let feed_forward_w3 = md_tensor(&format!("{prefix}.ffn_up.weight"), reader)?;
                FeedForwardVariant::Llama(LlamaFeedForward {
                    feed_forward_w1,
                    feed_forward_w2,
                    feed_forward_w3,
                })
            } else {
                // Otherwise, try to read from the up, and down weights
                let up = md_tensor(&format!("{prefix}.ffn_up.weight"), reader)?;
                // Transpose the down tensor
                let down = md_tensor(&format!("{prefix}.ffn_down.weight"), reader)?;
                let feed_forward_length: u32 = md_get(".feed_forward_length")?.try_into()?;
                let feed_forward_length = feed_forward_length as usize;

                FeedForwardVariant::Phi(PhiFeedForward {
                    up,
                    down,
                    feed_forward_length,
                })
            };
            let attention_norm = md_tensor(&format!("{prefix}.attn_norm.weight"), reader)?;
            let ffn_norm = md_tensor(&format!("{prefix}.ffn_norm.weight"), reader)?;
            layers.push(LlamaAttention {
                attention_variant,
                attention_wo,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps as f32)?,
                feed_forward_variant,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps as f32)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                hidden_size: config.hidden_size(),
                rope_cache: rope.clone(),
            })
        }
        Ok(Self {
            config,
            tok_embeddings: Embedding::new(tok_embeddings_q, embedding_length),
            layers,
            norm,
            output,
            masks: Default::default(),
        })
    }

    pub async fn forward(
        &self,
        tokens: &[u32],
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
    ) -> Tensor<1, f32> {
        let mut seq_len = tokens.len();
        let cached_tokens = cache.as_ref().map(|c| c.tokens.len()).unwrap_or_default();
        // We use a lower cutoff than the context length to avoid recomputing the attention every single token
        let cutoff_len: usize = self.config.context_length.saturating_sub(32).max(8);
        let (x, index_pos) = if seq_len + cached_tokens > self.config.context_length {
            let all_tokens = if let Some(cache) = cache.as_mut() {
                cache.clear();
                let mut all_tokens = cache.tokens.clone();
                all_tokens.extend(tokens);
                all_tokens
            } else {
                tokens.to_vec()
            };
            let start = all_tokens.len() - cutoff_len;
            seq_len = cutoff_len;
            tracing::trace!("The context is full, trimming start of the context to fit new tokens. The first {} tokens were truncated.", start);
            let all_tokens = &all_tokens[start..];
            if let Some(cache) = cache.as_mut() {
                cache.tokens = all_tokens.to_vec();
            }
            assert!(all_tokens.len() <= self.config.context_length);
            (Tensor::new(device, all_tokens), 0)
        } else {
            let index_pos = cache.as_ref().map(|c| c.tokens.len()).unwrap_or_default();
            if let Some(cache) = cache.as_mut() {
                cache.tokens.extend_from_slice(tokens);
            }
            (Tensor::new(device, tokens), index_pos)
        };
        let mask = self.masks.get_mask(seq_len, index_pos, device);

        let mut layer_in = self.tok_embeddings.forward(&x);
        for (i, layer) in self.layers.iter().enumerate() {
            let x = layer_in;
            let residual = x.clone();
            let x = layer.attention_norm.forward(&x);
            let attn = layer.forward(
                &x,
                Some(&mask),
                index_pos,
                cache.as_mut().map(|c| &mut c.blocks[i]),
            );
            let x = attn + residual;

            // MLP
            let residual = x.clone();
            let x = layer.ffn_norm.forward(&x);

            layer_in = layer.feed_forward_variant.forward(&x) + residual;
            // Materialize the layer output every layer to prevent the compute graph from growing too large
            let _ = layer_in.materialize();
        }
        let x = self.norm.forward(&layer_in);
        let [_, hidden_size] = x.shape();
        let x = x.slice([(seq_len - 1)..seq_len, 0..*hidden_size]);
        let out = x.q_mat_mul(&self.output);
        let [_, size] = *out.shape();
        out.reshape([size])
    }
}
