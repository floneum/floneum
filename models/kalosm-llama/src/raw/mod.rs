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
use candle_core::quantized::*;
use candle_core::IndexOp;
use candle_core::Module;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Embedding;
use candle_transformers::quantized_nn::RmsNorm;
use kalosm_common::MaskCache;

mod attention_layer;
pub mod cache;
mod rope;
mod silu;

use cache::LlamaCache;

fn decode_norm(tensor: QTensor, eps: f64) -> candle_core::Result<RmsNorm> {
    RmsNorm::from_qtensor(tensor, eps)
}

pub const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
pub const GEMMA_DEFAULT_SLIDING_WINDOW_TYPE: usize = 6;
pub const GEMMA_DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;

/// The configuration of a Llama model.
pub struct LlamaConfig {
    rope_freq_weight: Option<Tensor>,
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
    output: QMatMul,
    masks: MaskCache,
}

impl Model {
    pub fn from_ggml(
        mut ct: ggml_file::Content,
        gqa: usize,
        device: &Device,
        start_token_string: String,
        stop_token: u32,
        stop_token_string: String,
        rope_scaling: Option<RopeScalingConfig>,
    ) -> Result<Self> {
        let head_dim = (ct.hparams.n_embd / ct.hparams.n_head) as usize;
        let n_layer = ct.hparams.n_layer as usize;
        let config = LlamaConfig {
            rope_freq_weight: None,
            rope_theta: 10000.,
            head_dimension: head_dim,
            n_head: ct.hparams.n_head as usize,
            n_layer,
            context_length: 4096,
            start_token_string,
            stop_token,
            stop_token_string,
            chat_template: None,
            rope_scaling,
        };
        let config = Arc::new(config);
        let rope = RopeCache::new(&config, DType::F32, config.rope_theta, device)?;
        let tok_embeddings_q = ct.remove("tok_embeddings.weight")?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let output = if let Ok(output) = ct.remove("output.weight") {
            QMatMul::from_qtensor(output)?
        } else {
            // If there is no output layer, assume the word embeddings are tied to the output
            QMatMul::from_qtensor(tok_embeddings_q)?
        };
        let mut layers = Vec::with_capacity(n_layer);
        for layer_idx in 0..ct.hparams.n_layer {
            let prefix = format!("layers.{layer_idx}");
            let attention_wq = ct.remove(&format!("{prefix}.attention.wq.weight"))?;
            let attention_wk = ct.remove(&format!("{prefix}.attention.wk.weight"))?;
            let attention_wv = ct.remove(&format!("{prefix}.attention.wv.weight"))?;
            let attention_wo = ct.remove(&format!("{prefix}.attention.wo.weight"))?;
            let feed_forward_w1 = ct.remove(&format!("{prefix}.feed_forward.w1.weight"))?;
            let feed_forward_w2 = ct.remove(&format!("{prefix}.feed_forward.w2.weight"))?;
            let feed_forward_w3 = ct.remove(&format!("{prefix}.feed_forward.w3.weight"))?;
            let attention_norm = ct.remove(&format!("{prefix}.attention_norm.weight"))?;
            let ffn_norm = ct.remove(&format!("{prefix}.ffn_norm.weight"))?;
            let attention_variant = AttentionVariant::Separate(SeparateAttention {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_q_norm: None,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_k_norm: None,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                interleaved_rope: true,
                bias: None,
            });
            let feed_forward_variant = FeedForwardVariant::Llama(LlamaFeedForward {
                gate: QMatMul::from_qtensor(feed_forward_w1)?,
                up: QMatMul::from_qtensor(feed_forward_w2)?,
                down: QMatMul::from_qtensor(feed_forward_w3)?,
            });
            layers.push(LlamaAttention {
                attention_variant,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: decode_norm(attention_norm, 1e-5)?,
                post_attention_norm: None,
                feed_forward_variant,
                ffn_norm: decode_norm(ffn_norm, 1e-5)?,
                post_ffn_norm: None,
                n_head: ct.hparams.n_head as usize,
                n_kv_head: ct.hparams.n_head as usize / gqa,
                head_dim: (ct.hparams.n_embd / ct.hparams.n_head) as usize,
                hidden_size: config.hidden_size(),
                rope_cache: rope.clone(),
                sliding_window_size: None,
            })
        }

        Ok(Self {
            config,
            tok_embeddings: Embedding::new(tok_embeddings, ct.hparams.n_embd as usize),
            layers,
            norm: decode_norm(ct.remove("norm.weight")?, 1e-5)?,
            output,
            masks: Default::default(),
        })
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        override_stop_token_string: Option<String>,
        override_chat_template: Option<String>,
        rope_scaling: Option<RopeScalingConfig>,
    ) -> std::result::Result<Self, LlamaSourceError> {
        let md_get = |s: &str| {
            let value = if s.starts_with('.') {
                ct.metadata
                    .iter()
                    .filter(|(k, _)| k.ends_with(s))
                    .min_by_key(|(k, _)| k.len())
                    .map(|(_, v)| v)
            } else {
                ct.metadata.get(s)
            };
            match value {
                None => candle_core::bail!("cannot find {s} in metadata"),
                Some(v) => Ok(v),
            }
        };

        // Get the eos and bos tokens from the metadata
        let tokens: std::result::Result<Vec<_>, _> = md_get("tokenizer.ggml.tokens")?
            .to_vec()?
            .iter()
            .map(|v| v.to_string())
            .collect();
        let tokens = tokens?;
        let start_token = md_get("tokenizer.ggml.bos_token_id")
            .ok()
            .and_then(|v| v.to_u32().ok());
        let stop_token = if let Some(override_stop_token_string) = override_stop_token_string {
            tokens
                .iter()
                .position(|v| **v == override_stop_token_string)
                .unwrap_or(0) as u32
        } else {
            md_get("tokenizer.ggml.eos_token_id")?.to_u32()?
        };
        let start_token_string = start_token
            .map(|v| tokens[v as usize].clone())
            .unwrap_or_else(|| "".to_string());
        let stop_token_string = tokens[stop_token as usize].clone();
        let chat_template = override_chat_template.or_else(|| {
            md_get("tokenizer.chat_template")
                .ok()
                .and_then(|v| v.to_string().ok())
                .cloned()
        });
        let chat_template = match chat_template {
            Some(chat_template) => {
                let chat_template = HuggingFaceChatTemplate::create(chat_template)
                    .map_err(LlamaSourceError::ChatTemplate)?;
                Some(chat_template)
            }
            None => None,
        };

        // Parameter extraction from metadata.
        let architecture = ct.metadata["general.architecture"].to_string()?.clone();
        let head_count = md_get(".attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get(".attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get(".block_count")?.to_u32()? as usize;
        let embedding_length = md_get(".embedding_length")?.to_u32()? as usize;
        // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
        let rms_norm_eps = md_get(".attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let rope_freq_base = md_get(".rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);
        let sliding_window_size = md_get(".attention.sliding_window")
            .and_then(|m| m.to_u32())
            .ok()
            .map(|x| x as usize);
        let sliding_window_type = md_get(".attention.sliding_window_type")
            .and_then(|m| m.to_u32())
            .ok()
            .map(|x| x as usize)
            .or_else(|| (architecture == "gemma3").then_some(GEMMA_DEFAULT_SLIDING_WINDOW_TYPE));

        let rope_freq_base_sliding = md_get(".rope.local_freq_base")
            .and_then(|m| m.to_f32())
            .ok()
            .or_else(|| (architecture == "gemma3").then_some(GEMMA_DEFAULT_ROPE_FREQUENCY_SLIDING));

        let context_length = md_get(".context_length")?.to_u32()? as usize;
        let head_dim = md_get(".attention.key_length")
            .and_then(|v| v.to_u32())
            .ok()
            .map(|x| x as usize)
            .unwrap_or_else(|| embedding_length / head_count);

        let config = LlamaConfig {
            rope_freq_weight: match ct.tensor(reader, "rope_freqs.weight", device).ok() {
                Some(rope_freq_weight) => Some(rope_freq_weight.dequantize(device)?),
                None => None,
            },
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

        let rope = RopeCache::new(&config, DType::F32, config.rope_theta, device)?;
        let sliding_rope = rope_freq_base_sliding
            .map(|rope_freq_base_sliding| {
                RopeCache::new(&config, DType::F32, rope_freq_base_sliding, device)
            })
            .transpose()?;

        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let mut tok_embeddings = tok_embeddings_q.dequantize(device)?;
        // if this is gemma3, scale the tok_embeddings by sqrt(embedding_length)
        if architecture == "gemma3" {
            tok_embeddings = (tok_embeddings * (embedding_length as f64).sqrt())?;
        }
        let tok_embeddings = Embedding::new(tok_embeddings, embedding_length);

        let norm = ct.tensor(reader, "output_norm.weight", device)?;
        let norm = decode_norm(norm, rms_norm_eps)?;
        let output = if let Ok(output) = ct.tensor(reader, "output.weight", device) {
            QMatMul::from_qtensor(output)?
        } else {
            // If there is no output layer, assume the word embeddings are tied to the output
            QMatMul::from_qtensor(tok_embeddings_q)?
        };
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_variant =
                if let Ok(qkv) = ct.tensor(reader, &format!("{prefix}.attn_qkv.weight"), device) {
                    AttentionVariant::Grouped(GroupedAttention {
                        attention_qkv: QMatMul::from_qtensor(qkv)?,
                    })
                } else {
                    let q = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
                    let k = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
                    let v = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
                    let bias = if let (Ok(bias_q), Ok(bias_k), Ok(bias_v)) = (
                        ct.tensor(reader, &format!("{prefix}.attn_q.bias"), device),
                        ct.tensor(reader, &format!("{prefix}.attn_k.bias"), device),
                        ct.tensor(reader, &format!("{prefix}.attn_v.bias"), device),
                    ) {
                        Some(AttentionBias {
                            bias_q: bias_q.dequantize(device)?,
                            bias_k: bias_k.dequantize(device)?,
                            bias_v: bias_v.dequantize(device)?,
                        })
                    } else {
                        None
                    };
                    let q_norm = ct
                        .tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)
                        .ok();
                    let k_norm = ct
                        .tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)
                        .ok();
                    let separate = SeparateAttention {
                        attention_wq: QMatMul::from_qtensor(q)?,
                        attention_q_norm: q_norm
                            .map(|norm| decode_norm(norm, rms_norm_eps))
                            .transpose()?,
                        attention_wk: QMatMul::from_qtensor(k)?,
                        attention_k_norm: k_norm
                            .map(|norm| decode_norm(norm, rms_norm_eps))
                            .transpose()?,
                        attention_wv: QMatMul::from_qtensor(v)?,
                        interleaved_rope: architecture != "qwen2" && architecture != "gemma3",
                        bias,
                    };
                    AttentionVariant::Separate(separate)
                };
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            // Try to read from the up, down and gate weights
            let feed_forward_variant = if let Ok(ffn_gate) =
                ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)
            {
                let feed_forward_w1 = ffn_gate;
                let feed_forward_w2 =
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 =
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                FeedForwardVariant::Llama(LlamaFeedForward {
                    gate: QMatMul::from_qtensor(feed_forward_w1)?,
                    up: QMatMul::from_qtensor(feed_forward_w2)?,
                    down: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            } else {
                // Otherwise, try to read from the up, and down weights
                let up = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                // Transpose the down tensor
                let down = ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_length = md_get(".feed_forward_length")?.to_u32()? as usize;

                FeedForwardVariant::Phi(PhiFeedForward {
                    up: QMatMul::from_qtensor(up)?,
                    down: QMatMul::from_qtensor(down)?,
                    feed_forward_length,
                })
            };
            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let post_attention_norm = ct
                .tensor(
                    reader,
                    &format!("{prefix}.post_attention_norm.weight"),
                    device,
                )
                .ok();
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;
            let ffn_post_norm = ct
                .tensor(reader, &format!("{prefix}.post_ffw_norm.weight"), device)
                .ok();

            let mut layer_sliding_window_size = None;

            let rope_cache = if let (
                Some(rope_sliding),
                Some(sliding_window_type),
                Some(sliding_window_size),
            ) = (
                sliding_rope.as_ref(),
                sliding_window_type,
                sliding_window_size,
            ) {
                let is_sliding = (layer_idx + 1) % sliding_window_type != 0;
                if is_sliding {
                    layer_sliding_window_size = Some(sliding_window_size);
                    rope_sliding.clone()
                } else {
                    rope.clone()
                }
            } else {
                rope.clone()
            };

            layers.push(LlamaAttention {
                attention_variant,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: decode_norm(attention_norm, rms_norm_eps)?,
                post_attention_norm: post_attention_norm
                    .map(|norm| decode_norm(norm, rms_norm_eps))
                    .transpose()?,
                feed_forward_variant,
                ffn_norm: decode_norm(ffn_norm, rms_norm_eps)?,
                post_ffn_norm: ffn_post_norm
                    .map(|norm| decode_norm(norm, rms_norm_eps))
                    .transpose()?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                hidden_size: config.hidden_size(),
                rope_cache,
                sliding_window_size: layer_sliding_window_size,
            })
        }
        Ok(Self {
            config,
            tok_embeddings,
            layers,
            norm,
            output,
            masks: Default::default(),
        })
    }

    pub fn forward(
        &self,
        tokens: &[u32],
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
    ) -> Result<Tensor> {
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
            (Tensor::new(all_tokens, device)?.unsqueeze(0)?, 0)
        } else {
            let index_pos = cache.as_ref().map(|c| c.tokens.len()).unwrap_or_default();
            if let Some(cache) = cache.as_mut() {
                cache.tokens.extend_from_slice(tokens);
            }
            (Tensor::new(tokens, device)?.unsqueeze(0)?, index_pos)
        };

        let mut layer_in = self.tok_embeddings.forward(&x)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let x = layer_in;
            let residual = &x;
            debug_assert_none_nan(residual);
            let x = layer.attention_norm.forward(&x)?;
            debug_assert_none_nan(&x);
            let mask =
                self.masks
                    .get_mask(seq_len, index_pos, layer.sliding_window_size, device)?;
            let mut attn = layer.forward(
                &x,
                Some(&mask),
                index_pos,
                cache.as_mut().map(|c| &mut c.blocks[i]),
            )?;
            debug_assert_none_nan(&attn);
            if let Some(post_attention_norm) = &layer.post_attention_norm {
                attn = post_attention_norm.forward(&attn)?;
                debug_assert_none_nan(&attn);
            }
            let x = (attn + residual)?;
            debug_assert_none_nan(&x);

            // MLP
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            debug_assert_none_nan(&x);
            let mut x = layer.feed_forward_variant.forward(&x)?;
            debug_assert_none_nan(&x);
            if let Some(post_ffn_norm) = &layer.post_ffn_norm {
                x = post_ffn_norm.forward(&x)?;
                debug_assert_none_nan(&x);
            }

            layer_in = (&x + residual)?;
            debug_assert_none_nan(&layer_in);
        }
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        self.output.forward(&x)
    }
}

fn debug_assert_none_nan(tensor: &Tensor) {
    #[cfg(debug_assertions)]
    tensor
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
        .iter()
        .for_each(|v: &f32| {
            if v.is_nan() {
                panic!("Tensor contains NaN values");
            }
        });
}
