use crate::raw::attention_layer::LlamaAttention;
use crate::raw::rope::RopeCache;
use crate::raw::silu::fast_cpu_silu;
use candle_core::quantized::*;
use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::IndexOp;
use candle_core::Module;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Embedding;
use mask::MaskCache;

mod attention_layer;
pub mod cache;
mod mask;
mod rope;
mod silu;

use cache::LlamaCache;

fn decode_norm(tensor: QTensor, eps: f64) -> candle_core::Result<candle_nn::LayerNorm> {
    Ok(candle_nn::LayerNorm::rms_norm(
        tensor.dequantize(&Device::Cpu)?,
        eps,
    ))
}

#[allow(unused)]
pub(crate) struct LlamaConfig {
    rope_theta: f32,
    context_length: usize,
    head_dimension: usize,
    rope_dimension: usize,
    n_head: usize,
    n_kv_head: usize,
    pub(crate) n_layer: usize,
}

impl LlamaConfig {
    fn hidden_size(&self) -> usize {
        self.head_dimension * self.n_head
    }
}

pub struct Model {
    pub(crate) config: LlamaConfig,
    tok_embeddings: Embedding,
    layers: Vec<LlamaAttention>,
    norm: candle_nn::LayerNorm,
    output: QMatMul,
    masks: MaskCache,
}

impl Model {
    pub fn from_ggml(mut ct: ggml_file::Content, gqa: usize) -> anyhow::Result<Self> {
        let cpu = &Device::Cpu;
        let head_dim = (ct.hparams.n_embd / ct.hparams.n_head) as usize;
        let n_layer = ct.hparams.n_layer as usize;
        let config = LlamaConfig {
            rope_theta: 10000.,
            head_dimension: head_dim,
            rope_dimension: head_dim,
            n_head: ct.hparams.n_head as usize,
            n_kv_head: ct.hparams.n_head as usize / gqa,
            n_layer,
            context_length: 4096,
        };
        let rope = RopeCache::new(&config, DType::F32, cpu)?;
        let tok_embeddings = ct.remove("tok_embeddings.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(cpu)?;
        let output = ct.remove("output.weight")?;
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
            layers.push(LlamaAttention {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: decode_norm(attention_norm, 1e-5)?,
                feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
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
            output: QMatMul::from_qtensor(output)?,
            masks: Default::default(),
        })
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
    ) -> Result<Self> {
        let cpu = &Device::Cpu;
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Parameter extraction from metadata.
        let head_count = md_get("llama.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("llama.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("llama.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("llama.embedding_length")?.to_u32()? as usize;
        let rope_dim = md_get("llama.rope.dimension_count")?.to_u32()? as usize;
        // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
        let rms_norm_eps = md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let rope_freq_base = md_get("llama.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        let config = LlamaConfig {
            rope_theta: rope_freq_base,
            context_length: 4096,
            head_dimension: embedding_length / head_count,
            rope_dimension: rope_dim,
            n_head: head_count,
            n_kv_head: head_count_kv,
            n_layer: block_count,
        };

        let rope = RopeCache::new(&config, DType::F32, cpu)?;

        let tok_embeddings = ct.tensor(reader, "token_embd.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(cpu)?;
        let norm = decode_norm(ct.tensor(reader, "output_norm.weight")?, rms_norm_eps)?;
        let output = ct.tensor(reader, "output.weight")?;
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"))?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"))?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"))?;
            let attention_wo = ct.tensor(reader, &format!("{prefix}.attn_output.weight"))?;
            let feed_forward_w1 = ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"))?;
            let feed_forward_w2 = ct.tensor(reader, &format!("{prefix}.ffn_down.weight"))?;
            let feed_forward_w3 = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"))?;
            let attention_norm = ct.tensor(reader, &format!("{prefix}.attn_norm.weight"))?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"))?;
            layers.push(LlamaAttention {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: decode_norm(attention_norm, rms_norm_eps)?,
                feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                ffn_norm: decode_norm(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                hidden_size: config.hidden_size(),
                rope_cache: rope.clone(),
            })
        }
        Ok(Self {
            config,
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
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
        let (x, index_pos) = if seq_len + cached_tokens > cutoff_len {
            let all_tokens = if let Some(cache) = cache.as_mut() {
                cache.clear();
                let mut all_tokens = cache.tokens.clone();
                all_tokens.extend(tokens);
                all_tokens
            } else {
                tokens.to_vec()
            };
            let all_tokens = &all_tokens[all_tokens.len() - cutoff_len..];
            assert!(all_tokens.len() <= self.config.context_length);
            (Tensor::new(all_tokens, device)?.unsqueeze(0)?, 0)
        } else {
            let index_pos = cache.as_ref().map(|c| c.tokens.len()).unwrap_or_default();
            (Tensor::new(tokens, device)?.unsqueeze(0)?, index_pos)
        };
        if let Some(cache) = cache.as_mut() {
            cache.tokens.extend_from_slice(tokens);
        }
        let mask = self.masks.get_mask(seq_len, index_pos)?;

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

            layer_in = std::thread::scope(|scope| {
                let w1 = scope.spawn(|| {
                    let w1 = layer.feed_forward_w1.forward(&x)?;
                    fast_cpu_silu(&w1)
                });

                let w3 = layer.feed_forward_w3.forward(&x)?;
                let w1 = w1
                    .join()
                    .map_err(|_| candle_core::Error::Msg("Failed to join thread".to_string()))??;

                let mlp = layer.feed_forward_w2.forward(&(&w1 * w3)?)?;

                mlp + residual
            })?;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        self.output.forward(&x)
    }
}
