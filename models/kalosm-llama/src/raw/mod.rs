use std::sync::Arc;

use crate::raw::attention_layer::LlamaAttention;
use crate::raw::rope::RopeCache;
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

/// The configuration of a Llama model.
pub struct LlamaConfig {
    rope_freq_weight: Option<Tensor>,
    rope_theta: f32,
    pub(crate) context_length: usize,
    head_dimension: usize,
    n_head: usize,
    pub(crate) n_layer: usize,
    pub(crate) stop_token: u32,
}

impl LlamaConfig {
    fn hidden_size(&self) -> usize {
        self.head_dimension * self.n_head
    }
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
        stop_token: u32,
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
            stop_token,
        };
        let config = Arc::new(config);
        let rope = RopeCache::new(&config, DType::F32, device)?;
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
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                interleaved_rope: true,
                bias: None,
            });
            let feed_forward_variant = FeedForwardVariant::Llama(LlamaFeedForward {
                feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
            });
            layers.push(LlamaAttention {
                attention_variant,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: decode_norm(attention_norm, 1e-5)?,
                feed_forward_variant,
                ffn_norm: decode_norm(ffn_norm, 1e-5)?,
                n_head: ct.hparams.n_head as usize,
                n_kv_head: ct.hparams.n_head as usize / gqa,
                head_dim: (ct.hparams.n_embd / ct.hparams.n_head) as usize,
                hidden_size: config.hidden_size(),
                rope_cache: rope.clone(),
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
    ) -> Result<Self> {
        let md_get = |s: &str| {
            let value = if s.starts_with('.') {
                ct.metadata
                    .iter()
                    .find_map(|(k, value)| k.ends_with(s).then_some(value))
            } else {
                ct.metadata.get(s)
            };
            match value {
                None => candle_core::bail!("cannot find {s} in metadata"),
                Some(v) => Ok(v),
            }
        };

        // Get the eos and bos tokens from the metadata
        let bos_token = md_get("tokenizer.ggml.bos_token_id")?.to_u32()?;
        let stop_token = md_get("tokenizer.ggml.eos_token_id")?.to_u32()?;
        // let tokens = md_get("tokenizer.ggml.tokens")?.to_vec()?;
        // let chat_template = md_get("tokenizer.chat_template")?.to_string();

        // Parameter extraction from metadata.
        let head_count = md_get(".attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get(".attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get(".block_count")?.to_u32()? as usize;
        let embedding_length = md_get(".embedding_length")?.to_u32()? as usize;
        // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
        let rms_norm_eps = md_get(".attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let rope_freq_base = md_get(".rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10_000f32);

        let context_length = md_get(".context_length")?.to_u32()? as usize;
        let head_dim = embedding_length / head_count;

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
            stop_token,
        };
        let config = Arc::new(config);

        let rope = RopeCache::new(&config, DType::F32, device)?;

        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;

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
                    let architecture = ct.metadata["general.architecture"].to_string().unwrap();
                    let separate = SeparateAttention {
                        attention_wq: QMatMul::from_qtensor(q)?,
                        attention_wk: QMatMul::from_qtensor(k)?,
                        attention_wv: QMatMul::from_qtensor(v)?,
                        interleaved_rope: architecture != "qwen2",
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
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
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
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;
            layers.push(LlamaAttention {
                attention_variant,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: decode_norm(attention_norm, rms_norm_eps)?,
                feed_forward_variant,
                ffn_norm: decode_norm(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                hidden_size: config.hidden_size(),
                rope_cache: rope.clone(),
            })
        }
        Ok(Self {
            config,
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
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
        let seq_len = tokens.len();
        let cached_tokens = cache.as_ref().map(|c| c.tokens.len()).unwrap_or_default();
        // We use a lower cutoff than the context length to avoid recomputing the attention every single token
        let cutoff_len: usize = self.config.context_length - 32;
        let (x, index_pos) = if seq_len + cached_tokens > self.config.context_length {
            let all_tokens = if let Some(cache) = cache.as_mut() {
                cache.clear();
                let mut all_tokens = cache.tokens.clone();
                all_tokens.extend(tokens);
                all_tokens
            } else {
                tokens.to_vec()
            };
            let all_tokens = &all_tokens[all_tokens.len() - cutoff_len..];
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
        let mask = self.masks.get_mask(seq_len, index_pos, device)?;

        let mut layer_in = self.tok_embeddings.forward(&x)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward(
                &x,
                Some(&mask),
                index_pos,
                cache.as_mut().map(|c| &mut c.blocks[i]),
            )?;
            let x = (attn + residual)?;

            // MLP
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;

            layer_in = (&layer.feed_forward_variant.forward(&x)? + residual)?;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        self.output.forward(&x)
    }
}
