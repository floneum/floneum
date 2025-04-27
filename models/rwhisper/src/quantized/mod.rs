// Modified from https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/whisper/quantized_model.rs

use std::{num::NonZeroUsize, sync::Arc};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Module};
use candle_transformers::{
    models::whisper::Config,
    quantized_nn::{layer_norm, linear, linear_no_bias, Embedding, Linear},
    quantized_var_builder::VarBuilder,
};
use kalosm_common::{AttentionMask, KvCache, MaskCache, TensorCache};
use timestamps::extract_timestamps;

pub(crate) mod timestamps;

fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = vb
        .get((out_channels, in_channels, kernel_size), "weight")?
        .dequantize(vb.device())?;
    let bias = vb.get(out_channels, "bias")?.dequantize(vb.device())?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

struct MultiHeadAttentionCache {
    kv_cache: KvCache,
}

impl MultiHeadAttentionCache {
    fn new(max_seq_len: usize) -> Self {
        Self {
            kv_cache: KvCache::new(1, max_seq_len),
        }
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L62
#[derive(Debug, Clone)]
struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
    softmax_span: tracing::Span,
    matmul_span: tracing::Span,
}

impl MultiHeadAttention {
    fn load(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let softmax_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-softmax");
        let matmul_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-matmul");
        let query = linear(n_state, n_state, vb.pp("q_proj"))?;
        let value = linear(n_state, n_state, vb.pp("v_proj"))?;
        let key = linear_no_bias(n_state, n_state, vb.pp("k_proj"))?;
        let out = linear(n_state, n_state, vb.pp("out_proj"))?;
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
            softmax_span,
            matmul_span,
        })
    }

    fn forward_kv(
        &self,
        x: &Tensor,
        cache: Option<&mut MultiHeadAttentionCache>,
    ) -> Result<(Tensor, Tensor)> {
        let key_states = self.key.forward(x)?;
        let value_states = self.value.forward(x)?;
        let (key_states, value_states) = match cache {
            None => (key_states, value_states),
            Some(cache) => cache.kv_cache.append(&key_states, &value_states)?,
        };
        Ok((key_states, value_states))
    }

    fn forward(
        &mut self,
        query: &Tensor,
        kv: (Tensor, Tensor),
        mask: Option<&AttentionMask>,
        attention_output: Option<&mut TensorCache>,
    ) -> Result<Tensor> {
        let query_states = self.query.forward(query)?;
        let (key_states, value_states) = &kv;
        let wv = self.qkv_attention(
            &query_states,
            key_states,
            value_states,
            mask,
            attention_output,
        )?;
        let out = self.out.forward(&wv)?;
        Ok(out)
    }

    fn reshape_head(&self, x: &Tensor) -> Result<Tensor> {
        let (n_batch, n_ctx, n_state) = x.dims3()?;
        let target_dims = &[n_batch, n_ctx, self.n_head, n_state / self.n_head];
        x.reshape(target_dims)?.transpose(1, 2)
    }

    fn qkv_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&AttentionMask>,
        attention_output: Option<&mut TensorCache>,
    ) -> Result<Tensor> {
        let (_, _, n_state) = q.dims3()?;
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);
        let q = (self.reshape_head(q)? * scale)?;
        let k = (self.reshape_head(k)?.transpose(2, 3)? * scale)?;
        let v = self.reshape_head(v)?.contiguous()?;
        let mut qk = {
            let _enter = self.matmul_span.enter();
            q.matmul(&k)?
        };
        if let Some(mask) = mask {
            mask.forward(&mut qk)?
        }
        if let Some(out) = attention_output {
            out.append(&qk).unwrap();
        }
        let w = {
            let _enter = self.softmax_span.enter();
            candle_nn::ops::softmax_last_dim(&qk)?
        };
        let wv = {
            let _enter = self.matmul_span.enter();
            w.matmul(&v)?
        }
        .transpose(1, 2)?
        .flatten_from(2)?;
        Ok(wv)
    }
}

struct ResidualAttentionBlockCache {
    attn: MultiHeadAttentionCache,
    feature_attn_cache: Option<(Tensor, Tensor)>,
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L111
#[derive(Debug, Clone)]
struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross_attn: Option<(MultiHeadAttention, LayerNorm)>,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    mlp_ln: LayerNorm,
    span: tracing::Span,
}

impl ResidualAttentionBlock {
    fn load(n_state: usize, n_head: usize, cross_attn: bool, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "residual-attn");
        let attn = MultiHeadAttention::load(n_state, n_head, vb.pp("self_attn"))?;
        let attn_ln = layer_norm(n_state, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let cross_attn = if cross_attn {
            let cross_attn = MultiHeadAttention::load(n_state, n_head, vb.pp("encoder_attn"))?;
            let cross_attn_ln = layer_norm(n_state, 1e-5, vb.pp("encoder_attn_layer_norm"))?;
            Some((cross_attn, cross_attn_ln))
        } else {
            None
        };
        let n_mlp = n_state * 4;
        let mlp_linear1 = linear(n_state, n_mlp, vb.pp("fc1"))?;
        let mlp_linear2 = linear(n_mlp, n_state, vb.pp("fc2"))?;
        let mlp_ln = layer_norm(n_state, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            attn,
            attn_ln,
            cross_attn,
            mlp_linear1,
            mlp_linear2,
            mlp_ln,
            span,
        })
    }

    fn forward(
        &mut self,
        audio_features_kv: Option<(Tensor, Tensor)>,
        x: &Tensor,
        mask: Option<&AttentionMask>,
        mut cache: Option<&mut ResidualAttentionBlockCache>,
        attention_output: Option<&mut TensorCache>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let attn_ln_x = self.attn_ln.forward(x)?;
        let kv = self
            .attn
            .forward_kv(&attn_ln_x, cache.as_mut().map(|cache| &mut cache.attn))?;
        let attn = self.attn.forward(&attn_ln_x, kv, mask, None)?;
        let mut x = (x + attn)?;
        if let (Some(kv), Some((attn, ln))) = (audio_features_kv, &mut self.cross_attn) {
            let ln_x = ln.forward(&x)?;
            x = (&x + attn.forward(&ln_x, kv, None, attention_output)?)?;
        }
        let mlp = x
            .apply(&self.mlp_ln)?
            .apply(&self.mlp_linear1)?
            .gelu()?
            .apply(&self.mlp_linear2)?;
        x + mlp
    }
}

fn sinusoids(length: usize, channels: usize, device: &Device) -> Result<Tensor> {
    let max_timescale = 10000f32;
    let log_timescale_increment = max_timescale.ln() / (channels / 2 - 1) as f32;
    let inv_timescales: Vec<_> = (0..channels / 2)
        .map(|i| (i as f32 * (-log_timescale_increment)).exp())
        .collect();
    let inv_timescales = Tensor::new(inv_timescales.as_slice(), device)?.unsqueeze(0)?;
    let arange = Tensor::arange(0, length as u32, device)?
        .to_dtype(DType::F32)?
        .unsqueeze(1)?;
    let sh = (length, channels / 2);
    let scaled_time = (arange.broadcast_as(sh)? * inv_timescales.broadcast_as(sh)?)?;
    let sincos = Tensor::cat(&[scaled_time.sin()?, scaled_time.cos()?], 1)?;
    Ok(sincos)
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L143
#[derive(Debug, Clone)]
pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
    span: tracing::Span,
    conv1_span: tracing::Span,
    conv2_span: tracing::Span,
}

impl AudioEncoder {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "audio-encoder");
        let conv1_span = tracing::span!(tracing::Level::TRACE, "conv1");
        let conv2_span = tracing::span!(tracing::Level::TRACE, "conv2");
        let n_state = cfg.d_model;
        let n_head = cfg.encoder_attention_heads;
        let n_ctx = cfg.max_source_positions;
        let cfg1 = Conv1dConfig {
            padding: 1,
            stride: 1,
            groups: 1,
            dilation: 1,
        };
        let cfg2 = Conv1dConfig {
            padding: 1,
            stride: 2,
            groups: 1,
            dilation: 1,
        };
        let conv1 = conv1d(cfg.num_mel_bins, n_state, 3, cfg1, vb.pp("conv1"))?;
        let conv2 = conv1d(n_state, n_state, 3, cfg2, vb.pp("conv2"))?;
        let positional_embedding = sinusoids(n_ctx, n_state, vb.device())?;
        let blocks = (0..cfg.encoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(n_state, n_head, false, vb.pp(format!("layers.{i}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let ln_post = layer_norm(n_state, 1e-5, vb.pp("layer_norm"))?;
        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            ln_post,
            conv1_span,
            conv2_span,
            span,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = {
            let _enter = self.conv1_span.enter();
            self.conv1.forward(x)?.gelu()?
        };
        let x = {
            let _enter = self.conv2_span.enter();
            self.conv2.forward(&x)?.gelu()?
        };
        let x = x.transpose(1, 2)?;
        let (_bsize, seq_len, _hidden) = x.dims3()?;
        let positional_embedding = self.positional_embedding.narrow(0, 0, seq_len)?;
        let mut x = x.broadcast_add(&positional_embedding)?;
        for block in self.blocks.iter_mut() {
            x = block.forward(None, &x, None, None, None)?
        }
        let x = self.ln_post.forward(&x)?;
        Ok(x)
    }
}

#[derive(Default)]
pub struct TextDecoderCache {
    tokens: Vec<u32>,
    blocks: Vec<ResidualAttentionBlockCache>,
}

impl TextDecoderCache {
    pub fn new() -> Self {
        Self::default()
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L176
#[derive(Debug, Clone)]
pub struct TextDecoder {
    token_embedding: Embedding,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm,
    max_target_positions: usize,
    mask_cache: Arc<MaskCache>,
    span: tracing::Span,
    span_final: tracing::Span,
}

impl TextDecoder {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "text-decoder");
        let span_final = tracing::span!(tracing::Level::TRACE, "text-decoder-final");
        let n_state = cfg.d_model;
        let n_head = cfg.decoder_attention_heads;
        let max_target_positions = cfg.max_target_positions;
        let token_embedding = Embedding::new(cfg.vocab_size, n_state, vb.pp("embed_tokens"))?;
        let positional_embedding = vb
            .get((max_target_positions, n_state), "embed_positions.weight")?
            .dequantize(vb.device())?;
        let blocks = (0..cfg.decoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(n_state, n_head, true, vb.pp(format!("layers.{i}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let ln = layer_norm(n_state, 1e-5, vb.pp("layer_norm"))?;
        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            max_target_positions,
            mask_cache: Default::default(),
            span,
            span_final,
        })
    }

    pub fn forward(
        &mut self,
        tokens: &[u32],
        audio_features: &Tensor,
        cache: &mut TextDecoderCache,
        mut attention_output: Option<&mut [TensorCache]>,
    ) -> Result<Tensor> {
        let index_pos = cache.tokens.len();
        cache.tokens.extend_from_slice(tokens);
        let seq_len = tokens.len();
        if index_pos + seq_len > self.max_target_positions {
            candle_core::bail!("exceeded max sequence length")
        }
        let device = audio_features.device();
        let mask = self.mask_cache.get_mask(seq_len, index_pos, device)?;
        let x = Tensor::new(tokens, device)?;
        // The model expects a batch dim but this inference loop does not handle
        // it so we add it at this point.
        let x = x.unsqueeze(0)?;

        let _enter = self.span.enter();
        let token_embedding = self.token_embedding.forward(&x)?;
        let positional_embedding = self.positional_embedding.narrow(0, index_pos, seq_len)?;
        let mut x = token_embedding.broadcast_add(&positional_embedding)?;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            if cache.blocks.len() <= i {
                cache.blocks.push(ResidualAttentionBlockCache {
                    attn: MultiHeadAttentionCache::new(self.max_target_positions),
                    feature_attn_cache: block
                        .cross_attn
                        .as_ref()
                        .and_then(|(attn, _)| attn.forward_kv(audio_features, None).ok()),
                });
            }
            let block_cache = &mut cache.blocks[i];
            let query = block_cache.feature_attn_cache.clone();
            let attention_output = attention_output.as_mut().map(|outputs| &mut outputs[i]);
            x = block.forward(query, &x, Some(&mask), Some(block_cache), attention_output)?;
        }
        self.ln.forward(&x)
    }

    pub fn final_linear(&self, x: &Tensor) -> Result<Tensor> {
        let b_size = x.dim(0)?;
        let w = self.token_embedding.embeddings().broadcast_left(b_size)?;
        let logits = {
            let _enter = self.span_final.enter();
            x.matmul(&w.t()?)?
        };
        Ok(logits)
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L221
#[derive(Debug, Clone)]
pub struct Whisper {
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
    pub config: Config,
}

impl Whisper {
    pub fn load(vb: &VarBuilder, config: Config) -> Result<Self> {
        let encoder = AudioEncoder::load(vb.pp("model.encoder"), &config)?;
        let decoder = TextDecoder::load(vb.pp("model.decoder"), &config)?;
        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    pub(crate) fn dtw_timestamps(
        attention_heads: Option<&'static [[usize; 2]]>,
        filter_width: NonZeroUsize,
        n_frames: usize,
        mask: Vec<Vec<bool>>,
        attention_output: &[TensorCache],
    ) -> Result<Vec<Vec<f32>>> {
        let Some(attention_heads) = attention_heads else {
            return Err(candle_core::Error::msg(
                "The attention heads for word-level timestamps are not available for this model",
            ));
        };

        let mut attention_output_tensor = Vec::new();
        for attn in attention_output {
            attention_output_tensor.push(attn.current_data()?.clone().unwrap());
        }

        extract_timestamps(
            attention_heads,
            &attention_output_tensor,
            filter_width,
            n_frames,
            mask,
        )
    }
}
