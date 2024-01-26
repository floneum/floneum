use crate::raw::attention_layer::LlamaAttention;
use crate::raw::rope::RopeCache;
use candle_core::quantized::*;
use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Embedding;
use std::collections::HashMap;
use std::sync::RwLock;

mod attention_layer;
mod cache;
mod mask;
mod rope;
mod silu;

fn decode_norm(tensor: QTensor, eps: f64) -> candle_core::Result<candle_nn::LayerNorm> {
    Ok(candle_nn::LayerNorm::rms_norm(
        tensor.dequantize(&Device::Cpu)?,
        eps,
    ))
}

struct LlamaConfig {
    rope_theta: f32,
    context_length: usize,
    head_dimension: usize,
    rope_dimension: usize,
    n_head: usize,
    n_kv_head: usize,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            rope_theta: 1e-5,
            context_length: 4096,
            head_dimension: 64,
            rope_dimension: 64,
            n_head: 16,
            n_kv_head: 4,
        }
    }
}

pub struct Llama {
    tok_embeddings: Embedding,
    layers: Vec<LlamaAttention>,
    norm: candle_nn::LayerNorm,
    output: QMatMul,
    masks: RwLock<HashMap<String, Tensor>>,
}

impl Llama {
    pub fn from_ggml(mut ct: ggml_file::Content, gqa: usize) -> anyhow::Result<Self> {
        let cpu = &Device::Cpu;
        let head_dim = (ct.hparams.n_embd / ct.hparams.n_head) as usize;
        let config = LlamaConfig {
            rope_theta: 10000.,
            head_dimension: head_dim,
            rope_dimension: head_dim,
            n_head: ct.hparams.n_head as usize,
            n_kv_head: ct.hparams.n_head as usize / gqa,
            ..Default::default()
        };
        let rope = RopeCache::new(&config, DType::F32, cpu)?;
        let tok_embeddings = ct.remove("tok_embeddings.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(cpu)?;
        let output = ct.remove("output.weight")?;
        let mut layers = Vec::with_capacity(ct.hparams.n_layer as usize);
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
                rope_cache: rope.clone(),
            })
        }
        Ok(Self {
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
        let rms_norm_eps = md_get("llama.attention.layer_norm_rms_epsilon")?.to_f64()?;

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
                rope_cache: rope.clone(),
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            masks: Default::default(),
        })
    }
}
