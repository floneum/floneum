// Modified from https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/whisper/quantized_model.rs

use std::{num::NonZeroUsize, sync::Arc};

use fusor::{
    cache::{AttentionMask, KvCache, MaskCache, TensorCache},
    layers::{Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear},
    Device, Error, Result, Tensor, VarBuilder,
};
use timestamps::extract_timestamps;

use crate::config::Config;

pub(crate) mod timestamps;

fn conv1d(
    config: Conv1dConfig,
    device: &Device,
    vb: &mut VarBuilder,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
) -> Result<Conv1d<crate::WhisperDType>> {
    let weight_2d: Tensor<2, crate::WhisperDType> = vb.get("weight", device)?.dequantize();
    // Reshape from [out_channels, in_channels*kernel_size] to [out_channels, in_channels, kernel_size]
    let weight: Tensor<3, crate::WhisperDType, _> = weight_2d.reshape([out_channels, in_channels, kernel_size]);

    // Debug: Check weight values after reshape
    static DEBUG_CONV_WEIGHT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let weight_count = DEBUG_CONV_WEIGHT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if weight_count < 2 {
        match &weight {
            fusor::Tensor::Cpu(w) => {
                let w_data = w.eval().inner().clone();
                let shape = weight.shape();
                eprintln!("DEBUG CONV{} WEIGHT (CPU): shape {:?}", weight_count, shape);
                eprintln!("  [0,0,0..3]: {:?}", (0..3).map(|i| w_data.get([0, 0, i])).collect::<Vec<_>>());
                eprintln!("  [0,1,0..3]: {:?}", (0..3).map(|i| w_data.get([0, 1, i])).collect::<Vec<_>>());
                eprintln!("  [1,0,0..3]: {:?}", (0..3).map(|i| w_data.get([1, 0, i])).collect::<Vec<_>>());
            }
            fusor::Tensor::Gpu(_) => {
                let w_clone = weight.clone();
                let shape_clone = weight.shape().clone();
                std::thread::spawn(move || {
                    pollster::block_on(async {
                        if let Ok(slice) = w_clone.as_slice().await {
                            eprintln!("DEBUG CONV{} WEIGHT (GPU): shape {:?}", weight_count, shape_clone);
                            eprintln!("  [0,0,0..3]: {:?}", (0..3).map(|i| slice[[0, 0, i]]).collect::<Vec<f32>>());
                            eprintln!("  [0,1,0..3]: {:?}", (0..3).map(|i| slice[[0, 1, i]]).collect::<Vec<f32>>());
                            eprintln!("  [1,0,0..3]: {:?}", (0..3).map(|i| slice[[1, 0, i]]).collect::<Vec<f32>>());
                        }
                    });
                }).join().ok();
            }
        }
    }

    let bias_2d: Tensor<2, crate::WhisperDType> = vb.get("bias", device)?.dequantize();
    // Squeeze to rank 1: assume shape is (1, out_channels) or (out_channels, 1)
    let bias: Tensor<1, crate::WhisperDType, _> = if bias_2d.shape()[0] == 1 {
        bias_2d.squeeze(0)
    } else {
        bias_2d.squeeze(1)
    };
    Ok(Conv1d::new(weight, Some(bias), config))
}

struct MultiHeadAttentionCache {
    kv_cache: KvCache<crate::WhisperDType>,
}

impl MultiHeadAttentionCache {
    fn new(max_seq_len: usize) -> Self {
        Self {
            kv_cache: KvCache::new(1, max_seq_len),
        }
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L62
struct MultiHeadAttention {
    query: Linear<crate::WhisperDType>,
    key: Linear<crate::WhisperDType>,
    value: Linear<crate::WhisperDType>,
    out: Linear<crate::WhisperDType>,
    n_head: usize,
    softmax_span: tracing::Span,
    matmul_span: tracing::Span,
}

impl MultiHeadAttention {
    fn load(n_head: usize, device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let softmax_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-softmax");
        let matmul_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-matmul");
        let query = Linear::load(device, &mut vb.pp("q_proj"))?;
        let value = Linear::load(device, &mut vb.pp("v_proj"))?;
        let key = Linear::load(device, &mut vb.pp("k_proj"))?;
        let out = Linear::load(device, &mut vb.pp("out_proj"))?;
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
        x: &Tensor<3, crate::WhisperDType>,
        cache: Option<&mut MultiHeadAttentionCache>,
    ) -> Result<(
        Tensor<3, crate::WhisperDType>,
        Tensor<3, crate::WhisperDType>,
    )> {
        let device = x.device();
        let key_states = self.key.forward(x);
        let value_states = self.value.forward(x);
        let (key_states, value_states) = match cache {
            None => (key_states, value_states),
            Some(cache) => {
                let (k, v) = cache
                    .kv_cache
                    .append(&device, &key_states.unsqueeze(2), &value_states.unsqueeze(2));
                (k.squeeze(2), v.squeeze(2))
            }
        };
        Ok((key_states, value_states))
    }

    fn forward(
        &mut self,
        query: &Tensor<3, crate::WhisperDType>,
        kv: (
            Tensor<3, crate::WhisperDType>,
            Tensor<3, crate::WhisperDType>,
        ),
        mask: Option<&AttentionMask<crate::WhisperDType>>,
        attention_output: Option<&mut TensorCache<4, crate::WhisperDType>>,
    ) -> Result<Tensor<3, crate::WhisperDType>> {
        let query_states = self.query.forward(query);
        let (key_states, value_states) = &kv;
        let wv = self.qkv_attention(
            &query_states,
            key_states,
            value_states,
            mask,
            attention_output,
        )?;
        Ok(self.out.forward(&wv))
    }

    fn reshape_head(&self, x: &Tensor<3, crate::WhisperDType>) -> Tensor<4, crate::WhisperDType> {
        let [n_batch, n_ctx, n_state] = x.shape();
        let target_dims = [n_batch, n_ctx, self.n_head, n_state / self.n_head];
        x.reshape(target_dims).transpose(1, 2)
    }

    fn qkv_attention(
        &self,
        q: &Tensor<3, crate::WhisperDType>,
        k: &Tensor<3, crate::WhisperDType>,
        v: &Tensor<3, crate::WhisperDType>,
        mask: Option<&AttentionMask<crate::WhisperDType>>,
        attention_output: Option<&mut TensorCache<4, crate::WhisperDType>>,
    ) -> Result<Tensor<3, crate::WhisperDType>> {
        let device = q.device();
        let [_, _, n_state] = q.shape();
        let scale = crate::WhisperDType::from(((n_state / self.n_head) as f32).powf(-0.25));
        let q = self.reshape_head(q).mul_scalar(scale);
        let k = self.reshape_head(k).transpose(2, 3).mul_scalar(scale);
        let v = self.reshape_head(v);

        // Debug: Check shapes before matmul
        static DEBUG_ATTN_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let count = DEBUG_ATTN_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count < 5 {
            eprintln!("DEBUG ATTN[{}]: q shape: {:?}, k shape: {:?}, v shape: {:?}", count, q.shape(), k.shape(), v.shape());
        }

        let mut qk = {
            let _enter = self.matmul_span.enter();
            q.mat_mul(&k)
        };

        if count < 5 {
            eprintln!("DEBUG ATTN[{}]: qk shape: {:?}", count, qk.shape());
        }

        if let Some(mask) = mask {
            mask.forward(&mut qk)
        }
        if let Some(out) = attention_output {
            out.append(&device, &qk);
        }
        let w = {
            let _enter = self.softmax_span.enter();
            qk.softmax_last_dim()
        };

        // Debug: Print attention weights for cross-attention specifically
        // Cross-attention has shape [batch, heads, decoder_seq, encoder_seq] = [1, 6, 2, 1500]
        if count >= 8 && count < 10 {
            if let fusor::Tensor::Cpu(w_cpu) = &w {
                let w_data = w_cpu.eval().inner().clone();
                let shape = w.shape();
                // For cross-attn: shape [1, 6, 2, 1500], print head 0's attention weights
                if shape[3] > 10 {  // This is cross-attention (encoder has many positions)
                    eprintln!("DEBUG CROSS_ATTN_WEIGHTS[{}]: shape {:?}", count, shape);
                    // Print attention pattern for decoder position 1 attending to encoder
                    let row_sum: f32 = (0..shape[3]).map(|j| w_data.get([0, 0, 1, j])).sum();
                    eprintln!("  row 1 sum: {:.4} (should be ~1.0)", row_sum);
                    // Find where the decoder is attending most
                    let mut max_pos = 0;
                    let mut max_val = 0.0f32;
                    for j in 0..shape[3] {
                        let v = w_data.get([0, 0, 1, j]);
                        if v > max_val {
                            max_val = v;
                            max_pos = j;
                        }
                    }
                    eprintln!("  row 1 max attention: pos {} = {:.4}", max_pos, max_val);
                    // Print first 10 and last 10 attention values
                    let first10: Vec<f32> = (0..10.min(shape[3])).map(|j| w_data.get([0, 0, 1, j])).collect();
                    eprintln!("  row 1 first 10 positions: {:?}", first10);
                }
            }
        } else if count >= 4 && count < 7 {
            // Decoder self-attention debug (keep existing)
            if let fusor::Tensor::Cpu(w_cpu) = &w {
                let w_data = w_cpu.eval().inner().clone();
                let shape = w.shape();
                eprintln!("DEBUG ATTN[{}]: Decoder self-attention weights (head 0):", count);
                if shape[2] <= 4 && shape[3] <= 4 {
                    for i in 0..shape[2] {
                        let row: Vec<f32> = (0..shape[3]).map(|j| w_data.get([0, 0, i, j])).collect();
                        eprintln!("  row {}: {:?}", i, row);
                    }
                }
            }

            // Also check value vectors for positions 0 and 1 (head 0)
            if let fusor::Tensor::Cpu(v_cpu) = &v {
                let v_data = v_cpu.eval().inner().clone();
                let v_shape = v.shape();
                // v shape: [1, 6, seq, 64]
                if v_shape[2] >= 2 {
                    eprintln!("DEBUG ATTN[{}]: Value vectors (head 0, first 5 dims):", count);
                    let v0: Vec<f32> = (0..5.min(v_shape[3])).map(|i| v_data.get([0, 0, 0, i])).collect();
                    let v1: Vec<f32> = (0..5.min(v_shape[3])).map(|i| v_data.get([0, 0, 1, i])).collect();
                    eprintln!("  v[pos=0]: {:?}", v0);
                    eprintln!("  v[pos=1]: {:?}", v1);
                }
            }
        }

        if count < 5 {
            eprintln!("DEBUG ATTN[{}]: w (softmax) shape: {:?}", count, w.shape());
        }

        let wv_raw = {
            let _enter = self.matmul_span.enter();
            w.mat_mul(&v)
        };

        // Debug: Check wv before transpose for decoder attention
        if count >= 4 && count < 7 {
            if let fusor::Tensor::Cpu(wv_cpu) = &wv_raw {
                let wv_data = wv_cpu.eval().inner().clone();
                let wv_shape = wv_raw.shape();
                // wv shape: [1, 6, seq, 64]
                if wv_shape[2] >= 2 {
                    eprintln!("DEBUG ATTN[{}]: wv (before transpose, head 0, first 5 dims):", count);
                    let wv0: Vec<f32> = (0..5.min(wv_shape[3])).map(|i| wv_data.get([0, 0, 0, i])).collect();
                    let wv1: Vec<f32> = (0..5.min(wv_shape[3])).map(|i| wv_data.get([0, 0, 1, i])).collect();
                    eprintln!("  wv[pos=0]: {:?}", wv0);
                    eprintln!("  wv[pos=1]: {:?}", wv1);
                }
            }
        }

        let wv = wv_raw
        .transpose(1, 2)
        .flatten_last_n::<1, _>();

        if count < 5 {
            eprintln!("DEBUG ATTN[{}]: wv output shape: {:?}", count, wv.shape());
        }

        // Debug: Check wv after transpose and flatten for decoder attention
        if count >= 4 && count < 7 {
            if let fusor::Tensor::Cpu(wv_cpu) = &wv {
                let wv_data = wv_cpu.eval().inner().clone();
                let wv_shape = wv.shape();
                // wv shape: [1, seq, hidden]
                if wv_shape[1] >= 2 {
                    eprintln!("DEBUG ATTN[{}]: wv (after transpose/flatten, first 5 dims):", count);
                    let wv0: Vec<f32> = (0..5.min(wv_shape[2])).map(|i| wv_data.get([0, 0, i])).collect();
                    let wv1: Vec<f32> = (0..5.min(wv_shape[2])).map(|i| wv_data.get([0, 1, i])).collect();
                    eprintln!("  wv[pos=0]: {:?}", wv0);
                    eprintln!("  wv[pos=1]: {:?}", wv1);
                }
            }
        }

        Ok(wv)
    }
}

struct ResidualAttentionBlockCache {
    attn: MultiHeadAttentionCache,
    feature_attn_cache: Option<(
        Tensor<3, crate::WhisperDType>,
        Tensor<3, crate::WhisperDType>,
    )>,
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L111
struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm<1, crate::WhisperDType>,
    cross_attn: Option<(MultiHeadAttention, LayerNorm<1, crate::WhisperDType>)>,
    mlp_linear1: Linear<crate::WhisperDType>,
    mlp_linear2: Linear<crate::WhisperDType>,
    mlp_ln: LayerNorm<1, crate::WhisperDType>,
    span: tracing::Span,
}

impl ResidualAttentionBlock {
    fn load(n_head: usize, cross_attn: bool, device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "residual-attn");
        let attn = MultiHeadAttention::load(n_head, device, &mut vb.pp("self_attn"))?;
        let attn_ln = LayerNorm::load(device, &mut vb.pp("self_attn_layer_norm"), 1e-5)?;
        let cross_attn = if cross_attn {
            let cross_attn = MultiHeadAttention::load(n_head, device, &mut vb.pp("encoder_attn"))?;
            let cross_attn_ln =
                LayerNorm::load(device, &mut vb.pp("encoder_attn_layer_norm"), 1e-5)?;
            Some((cross_attn, cross_attn_ln))
        } else {
            None
        };
        let mlp_linear1 = Linear::load(device, &mut vb.pp("fc1"))?;
        let mlp_linear2 = Linear::load(device, &mut vb.pp("fc2"))?;
        let mlp_ln = LayerNorm::load(device, &mut vb.pp("final_layer_norm"), 1e-5)?;
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
        audio_features_kv: Option<(
            Tensor<3, crate::WhisperDType>,
            Tensor<3, crate::WhisperDType>,
        )>,
        x: &Tensor<3, crate::WhisperDType>,
        mask: Option<&AttentionMask<crate::WhisperDType>>,
        mut cache: Option<&mut ResidualAttentionBlockCache>,
        attention_output: Option<&mut TensorCache<4, crate::WhisperDType>>,
    ) -> Result<Tensor<3, crate::WhisperDType>> {
        let _enter = self.span.enter();

        let attn_ln_x = self.attn_ln.forward(x);
        let kv = self
            .attn
            .forward_kv(&attn_ln_x, cache.as_mut().map(|cache| &mut cache.attn))?;
        let attn = self.attn.forward(&attn_ln_x, kv, mask, None)?;
        let mut x = (x + &attn).to_concrete();

        // Debug: Check if cross-attention is being applied
        static DEBUG_CROSS_ATTN_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let cross_attn_count = DEBUG_CROSS_ATTN_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Debug: Check hidden state after self-attention residual
        if cross_attn_count < 8 {
            if let fusor::Tensor::Cpu(x_cpu) = &x {
                let x_data = x_cpu.eval().inner().clone();
                let x_shape = x.shape();
                if x_shape[1] >= 2 {
                    eprintln!("DEBUG BLOCK[{}]: After self-attn+residual (first 5 dims):", cross_attn_count);
                    let x0: Vec<f32> = (0..5.min(x_shape[2])).map(|i| x_data.get([0, 0, i])).collect();
                    let x1: Vec<f32> = (0..5.min(x_shape[2])).map(|i| x_data.get([0, 1, i])).collect();
                    eprintln!("  x[pos=0]: {:?}", x0);
                    eprintln!("  x[pos=1]: {:?}", x1);
                }
            }
        }

        if let (Some(kv), Some((attn, ln))) = (audio_features_kv, &mut self.cross_attn) {
            if cross_attn_count < 5 {
                eprintln!("DEBUG CROSS_ATTN[{}]: Applying cross-attention. kv key shape: {:?}, value shape: {:?}",
                         cross_attn_count, kv.0.shape(), kv.1.shape());
            }
            let ln_x = ln.forward(&x);
            let attn_out = attn.forward(&ln_x, kv, None, attention_output)?;
            if cross_attn_count < 5 {
                eprintln!("DEBUG CROSS_ATTN[{}]: attn_out shape: {:?}", cross_attn_count, attn_out.shape());
            }
            x = (&x + &attn_out).to_concrete();

            // Debug: Check hidden state after cross-attention residual
            if cross_attn_count < 8 {
                if let fusor::Tensor::Cpu(x_cpu) = &x {
                    let x_data = x_cpu.eval().inner().clone();
                    let x_shape = x.shape();
                    if x_shape[1] >= 2 {
                        eprintln!("DEBUG BLOCK[{}]: After cross-attn+residual (first 5 dims):", cross_attn_count);
                        let x0: Vec<f32> = (0..5.min(x_shape[2])).map(|i| x_data.get([0, 0, i])).collect();
                        let x1: Vec<f32> = (0..5.min(x_shape[2])).map(|i| x_data.get([0, 1, i])).collect();
                        eprintln!("  x[pos=0]: {:?}", x0);
                        eprintln!("  x[pos=1]: {:?}", x1);
                    }
                }
            }
        }
        let mlp = self
            .mlp_linear2
            .forward(&self.mlp_linear1.forward(&self.mlp_ln.forward(&x)).gelu());
        let result = (x + mlp).to_concrete();

        // Debug: Check hidden state after MLP+residual
        if cross_attn_count < 8 {
            if let fusor::Tensor::Cpu(r_cpu) = &result {
                let r_data = r_cpu.eval().inner().clone();
                let r_shape = result.shape();
                if r_shape[1] >= 2 {
                    eprintln!("DEBUG BLOCK[{}]: After MLP+residual (first 5 dims):", cross_attn_count);
                    let r0: Vec<f32> = (0..5.min(r_shape[2])).map(|i| r_data.get([0, 0, i])).collect();
                    let r1: Vec<f32> = (0..5.min(r_shape[2])).map(|i| r_data.get([0, 1, i])).collect();
                    eprintln!("  result[pos=0]: {:?}", r0);
                    eprintln!("  result[pos=1]: {:?}", r1);
                }
            }
        }

        Ok(result)
    }
}

fn sinusoids(length: usize, channels: usize, device: &Device) -> Tensor<2, crate::WhisperDType> {
    let max_timescale = 10000f32;
    let log_timescale_increment = crate::WhisperDType::from(max_timescale.ln())
        / crate::WhisperDType::from((channels / 2 - 1) as f32);
    let inv_timescales: Vec<_> = (0..channels / 2)
        .map(|i| (crate::WhisperDType::from(i as f32) * (-log_timescale_increment)).exp())
        .collect();
    let inv_timescales = Tensor::new(device, inv_timescales.as_slice()).unsqueeze(0);
    let arange = fusor::arange(device, 0u32, length as u32)
        .cast::<crate::WhisperDType>()
        .unsqueeze(1);
    let sh = [length, channels / 2];
    let scaled_time = (&arange.broadcast_as(sh) * &inv_timescales.broadcast_as(sh)).to_concrete();
    Tensor::cat([scaled_time.sin(), scaled_time.cos()], 1)
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L143
pub struct AudioEncoder {
    conv1: Conv1d<crate::WhisperDType>,
    conv2: Conv1d<crate::WhisperDType>,
    positional_embedding: Tensor<2, crate::WhisperDType>,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm<1, crate::WhisperDType>,
    span: tracing::Span,
    conv1_span: tracing::Span,
    conv2_span: tracing::Span,
}

impl AudioEncoder {
    fn load(device: &Device, mut vb: VarBuilder, cfg: &Config) -> Result<Self> {
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
        let n_mels = cfg.num_mel_bins;
        let conv1 = conv1d(cfg1, device, &mut vb.pp("conv1"), n_mels, n_state, 3)?;
        let conv2 = conv1d(cfg2, device, &mut vb.pp("conv2"), n_state, n_state, 3)?;
        let positional_embedding = sinusoids(n_ctx, n_state, device);
        let blocks = (0..cfg.encoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(
                    n_head,
                    false,
                    device,
                    &mut vb.pp(format!("layers.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let ln_post = LayerNorm::load(device, &mut vb.pp("layer_norm"), 1e-5)?;
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

    pub fn forward(
        &mut self,
        x: &Tensor<3, crate::WhisperDType>,
    ) -> Result<Tensor<3, crate::WhisperDType>> {
        let _enter = self.span.enter();

        eprintln!("DEBUG ENCODER: Input shape: {:?}", x.shape());

        // Debug: Check mel input values
        static DEBUG_MEL_INPUT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_MEL_INPUT.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let shape = x.shape();
            match x {
                fusor::Tensor::Cpu(ref cpu_t) => {
                    let data = cpu_t.eval().inner().clone();
                    eprintln!("DEBUG MEL INPUT (CPU): shape {:?}", shape);
                    eprintln!("  [0,0,0..10]: {:?}", (0..10).map(|i| data.get([0, 0, i])).collect::<Vec<_>>());
                    eprintln!("  [0,1,0..10]: {:?}", (0..10).map(|i| data.get([0, 1, i])).collect::<Vec<_>>());
                    eprintln!("  [0,40,0..10]: {:?}", (0..10).map(|i| data.get([0, 40, i])).collect::<Vec<_>>());
                    // Calculate stats
                    let mut min_v = f32::INFINITY;
                    let mut max_v = f32::NEG_INFINITY;
                    let mut sum_v = 0.0f32;
                    let mut count = 0;
                    for ch in 0..shape[1].min(10) {
                        for pos in 0..shape[2].min(100) {
                            let v = data.get([0, ch, pos]);
                            min_v = min_v.min(v);
                            max_v = max_v.max(v);
                            sum_v += v;
                            count += 1;
                        }
                    }
                    eprintln!("DEBUG MEL INPUT (CPU): stats (first 10x100): min={:.4}, max={:.4}, mean={:.4}", min_v, max_v, sum_v / count as f32);
                }
                fusor::Tensor::Gpu(_) => {
                    let x_clone = x.clone();
                    let shape_clone = shape.clone();
                    std::thread::spawn(move || {
                        pollster::block_on(async {
                            if let Ok(slice) = x_clone.as_slice().await {
                                eprintln!("DEBUG MEL INPUT (GPU): shape {:?}", shape_clone);
                                eprintln!("  [0,0,0..5]: {:?}", (0..5).map(|i| slice[[0, 0, i]]).collect::<Vec<f32>>());
                                eprintln!("  [0,40,0..5]: {:?}", (0..5).map(|i| slice[[0, 40, i]]).collect::<Vec<f32>>());
                                // Calculate stats
                                let mut min_v = f32::INFINITY;
                                let mut max_v = f32::NEG_INFINITY;
                                let mut sum_v = 0.0f32;
                                let mut count = 0;
                                for ch in 0..shape_clone[1].min(10) {
                                    for pos in 0..shape_clone[2].min(100) {
                                        let v = slice[[0, ch, pos]];
                                        min_v = min_v.min(v);
                                        max_v = max_v.max(v);
                                        sum_v += v;
                                        count += 1;
                                    }
                                }
                                eprintln!("DEBUG MEL INPUT (GPU): stats (first 10x100): min={:.4}, max={:.4}, mean={:.4}", min_v, max_v, sum_v / count as f32);
                            }
                        });
                    }).join().ok();
                }
            }
        }

        let x = {
            let _enter = self.conv1_span.enter();
            let conv_out = self.conv1.forward(x);
            eprintln!("DEBUG ENCODER: After conv1 shape: {:?}", conv_out.shape());

            // Debug: Check conv1 output values (before GELU)
            static DEBUG_CONV1: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !DEBUG_CONV1.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let conv_shape = conv_out.shape();
                match &conv_out {
                    fusor::Tensor::Cpu(cpu_t) => {
                        let data = cpu_t.eval().inner().clone();
                        eprintln!("DEBUG CONV1 OUTPUT (CPU): first values [0,0,0..5]: {:?}",
                            (0..5).map(|i| data.get([0, 0, i])).collect::<Vec<_>>());
                        eprintln!("DEBUG CONV1 OUTPUT (CPU): [0,100,0..5]: {:?}",
                            (0..5).map(|i| data.get([0, 100, i])).collect::<Vec<_>>());
                    }
                    fusor::Tensor::Gpu(_) => {
                        let conv_clone = conv_out.clone();
                        let conv_shape_clone = conv_shape.clone();
                        std::thread::spawn(move || {
                            pollster::block_on(async {
                                if let Ok(slice) = conv_clone.as_slice().await {
                                    eprintln!("DEBUG CONV1 OUTPUT (GPU): first values [0,0,0..5]: {:?}",
                                        (0..5).map(|i| slice[[0, 0, i]]).collect::<Vec<f32>>());
                                    eprintln!("DEBUG CONV1 OUTPUT (GPU): [0,100,0..5]: {:?}",
                                        (0..5).map(|i| slice[[0, 100, i]]).collect::<Vec<f32>>());
                                }
                            });
                        }).join().ok();
                    }
                }
            }

            conv_out.gelu()
        };
        let x = {
            let _enter = self.conv2_span.enter();
            let conv_out = self.conv2.forward(&x);
            eprintln!("DEBUG ENCODER: After conv2 shape: {:?}", conv_out.shape());
            conv_out.gelu()
        };
        let x = x.transpose(1, 2);
        let [_bsize, seq_len, _hidden] = x.shape();
        eprintln!("DEBUG ENCODER: After transpose shape: {:?}", x.shape());

        let positional_embedding = self.positional_embedding.narrow(0, 0, seq_len);
        let mut x = x.add_(&positional_embedding);
        eprintln!("DEBUG ENCODER: After positional embedding, starting {} blocks", self.blocks.len());

        for (block_idx, block) in self.blocks.iter_mut().enumerate() {
            x = block.forward(None, &x, None, None, None)?;
            if block_idx == 0 {
                eprintln!("DEBUG ENCODER: After block 0 shape: {:?}", x.shape());
            }
        }
        let x = self.ln_post.forward(&x);
        eprintln!("DEBUG ENCODER: Final output shape: {:?}", x.shape());

        // Debug: check encoder output values (works for both CPU and GPU)
        static DEBUG_ENCODER_OUTPUT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_ENCODER_OUTPUT.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let shape = x.shape();
            match &x {
                fusor::Tensor::Cpu(ref cpu_tensor) => {
                    let concrete = cpu_tensor.eval();
                    let inner = concrete.inner().clone();
                    // Sample some values from different positions
                    eprintln!("DEBUG ENCODER OUTPUT (CPU): sample values at positions 0, 100, 500, 1000:");
                    for pos in [0, 100, 500, 1000] {
                        if pos < shape[1] {
                            let vals: Vec<f32> = (0..5).map(|i| inner.get([0, pos, i])).collect();
                            eprintln!("  pos {}: {:?}", pos, vals);
                        }
                    }
                    // Calculate stats
                    let mut min_v = f32::INFINITY;
                    let mut max_v = f32::NEG_INFINITY;
                    let mut sum_v = 0.0f32;
                    let mut count = 0;
                    for pos in 0..shape[1].min(100) {
                        for dim in 0..shape[2].min(50) {
                            let v = inner.get([0, pos, dim]);
                            min_v = min_v.min(v);
                            max_v = max_v.max(v);
                            sum_v += v;
                            count += 1;
                        }
                    }
                    eprintln!("DEBUG ENCODER OUTPUT (CPU): stats (first 100x50): min={:.4}, max={:.4}, mean={:.4}", min_v, max_v, sum_v / count as f32);
                }
                fusor::Tensor::Gpu(_) => {
                    // Use pollster to block on GPU async operations for debug
                    let x_clone = x.clone();
                    let shape_clone = shape.clone();
                    std::thread::spawn(move || {
                        pollster::block_on(async {
                            if let Ok(slice) = x_clone.as_slice().await {
                                eprintln!("DEBUG ENCODER OUTPUT (GPU): sample values at positions 0, 100, 500, 1000:");
                                for pos in [0usize, 100, 500, 1000] {
                                    if pos < shape_clone[1] {
                                        let vals: Vec<f32> = (0..5).map(|i| slice[[0, pos, i]]).collect();
                                        eprintln!("  pos {}: {:?}", pos, vals);
                                    }
                                }
                                // Calculate stats
                                let mut min_v = f32::INFINITY;
                                let mut max_v = f32::NEG_INFINITY;
                                let mut sum_v = 0.0f32;
                                let mut count = 0;
                                for pos in 0..shape_clone[1].min(100) {
                                    for dim in 0..shape_clone[2].min(50) {
                                        let v = slice[[0, pos, dim]];
                                        min_v = min_v.min(v);
                                        max_v = max_v.max(v);
                                        sum_v += v;
                                        count += 1;
                                    }
                                }
                                eprintln!("DEBUG ENCODER OUTPUT (GPU): stats (first 100x50): min={:.4}, max={:.4}, mean={:.4}", min_v, max_v, sum_v / count as f32);
                            }
                        });
                    }).join().ok();
                }
            }
        }

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
pub struct TextDecoder {
    token_embedding: Embedding<crate::WhisperDType>,
    positional_embedding: Tensor<2, crate::WhisperDType>,
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm<1, crate::WhisperDType>,
    max_target_positions: usize,
    mask_cache: Arc<MaskCache<crate::WhisperDType>>,
    span: tracing::Span,
    span_final: tracing::Span,
}

impl TextDecoder {
    fn load(device: &Device, vb: &mut VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "text-decoder");
        let span_final = tracing::span!(tracing::Level::TRACE, "text-decoder-final");
        let n_head = cfg.decoder_attention_heads;
        let max_target_positions = cfg.max_target_positions;
        let token_embedding = Embedding::load(device, &mut vb.pp("embed_tokens"))?;
        let positional_embedding = vb.get("embed_positions.weight", device)?.dequantize();
        let blocks = (0..cfg.decoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(
                    n_head,
                    true,
                    device,
                    &mut vb.pp(format!("layers.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let ln = LayerNorm::load(device, &mut vb.pp("layer_norm"), 1e-5)?;
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
        audio_features: &Tensor<2, crate::WhisperDType>,
        cache: &mut TextDecoderCache,
        mut attention_output: Option<&mut [TensorCache<4, crate::WhisperDType>]>,
    ) -> Result<Tensor<3, crate::WhisperDType>> {
        let index_pos = cache.tokens.len();
        cache.tokens.extend_from_slice(tokens);
        let seq_len = tokens.len();
        eprintln!("DEBUG DECODER forward: tokens={:?}, index_pos={}, cache_len={}", tokens, index_pos, cache.tokens.len());
        if index_pos + seq_len > self.max_target_positions {
            return Err(Error::msg("exceeded max sequence length"));
        }
        let device = audio_features.device();
        let mask = self.mask_cache.get_mask(seq_len, index_pos, None, &device);
        let x: Tensor<1, u32> = Tensor::new(&device, tokens);
        // The model expects a batch dim but this inference loop does not handle
        // it so we add it at this point.
        let x = x.unsqueeze(0);

        let _enter = self.span.enter();
        let token_embedding = self.token_embedding.forward(&x);
        let positional_embedding = self.positional_embedding.narrow(0, index_pos, seq_len);
        // Debug: Check positional embedding values (only on first call)
        if index_pos == 0 && seq_len == 2 {
            eprintln!("DEBUG DECODER: Positional embedding first 5 values for position 0:");
            if let Tensor::Cpu(pos_emb) = &self.positional_embedding {
                let pos_eval = pos_emb.eval();
                let pos_data = pos_eval.inner();
                for j in 0..5 {
                    eprintln!("  pos_emb[0, {}] = {}", j, pos_data.get([0, j]));
                }
            }
        }
        eprintln!("DEBUG DECODER: token_embedding shape: {:?}, positional_embedding shape: {:?}", token_embedding.shape(), positional_embedding.shape());

        // Debug: Check if token embeddings are different for different positions
        static DEBUG_EMBED_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let embed_count = DEBUG_EMBED_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if embed_count < 3 && seq_len >= 2 {
            if let Tensor::Cpu(tok_emb) = &token_embedding {
                let tok_data = tok_emb.eval().inner().clone();
                eprintln!("DEBUG EMBED[{}]: Token embeddings at positions 0 and 1 (first 5 values):", embed_count);
                eprintln!("  tok_emb[0, 0, 0..5] = {:?}", (0..5).map(|i| tok_data.get([0, 0, i])).collect::<Vec<_>>());
                eprintln!("  tok_emb[0, 1, 0..5] = {:?}", (0..5).map(|i| tok_data.get([0, 1, i])).collect::<Vec<_>>());
            }
            if let Tensor::Cpu(pos_emb) = &positional_embedding {
                let pos_data = pos_emb.eval().inner().clone();
                eprintln!("DEBUG EMBED[{}]: Positional embeddings at positions 0 and 1 (first 5 values):", embed_count);
                eprintln!("  pos_emb[0, 0..5] = {:?}", (0..5).map(|i| pos_data.get([0, i])).collect::<Vec<_>>());
                eprintln!("  pos_emb[1, 0..5] = {:?}", (0..5).map(|i| pos_data.get([1, i])).collect::<Vec<_>>());
            }
        }

        let mut x = token_embedding.add_(&positional_embedding);
        eprintln!("DEBUG DECODER: after embedding add shape: {:?}", x.shape());

        // Debug: Check combined embeddings after adding
        if embed_count < 3 && seq_len >= 2 {
            if let Tensor::Cpu(combined) = &x {
                let comb_data = combined.eval().inner().clone();
                eprintln!("DEBUG EMBED[{}]: Combined (tok+pos) at positions 0 and 1 (first 5 values):", embed_count);
                eprintln!("  combined[0, 0, 0..5] = {:?}", (0..5).map(|i| comb_data.get([0, 0, i])).collect::<Vec<_>>());
                eprintln!("  combined[0, 1, 0..5] = {:?}", (0..5).map(|i| comb_data.get([0, 1, i])).collect::<Vec<_>>());
            }
        }
        // Add batch dimension to audio_features for forward_kv
        let audio_features_batched = audio_features.unsqueeze(0);

        // Debug: Check audio features being passed to cross-attention (first time only)
        if index_pos == 0 && seq_len == 2 {
            eprintln!("DEBUG DECODER: audio_features_batched shape for cross-attn: {:?}", audio_features_batched.shape());
        }

        for (i, block) in self.blocks.iter_mut().enumerate() {
            if cache.blocks.len() <= i {
                // Debug: First block cross-attention cache creation
                if i == 0 && index_pos == 0 {
                    eprintln!("DEBUG DECODER: Creating cross-attn cache for block 0");
                }
                cache.blocks.push(ResidualAttentionBlockCache {
                    attn: MultiHeadAttentionCache::new(self.max_target_positions),
                    feature_attn_cache: block
                        .cross_attn
                        .as_ref()
                        .and_then(|(attn, _)| attn.forward_kv(&audio_features_batched, None).ok()),
                });
            }
            let block_cache = &mut cache.blocks[i];
            let query = block_cache.feature_attn_cache.clone();
            let attention_output = attention_output.as_mut().map(|outputs| &mut outputs[i]);
            x = block.forward(query, &x, Some(&mask), Some(block_cache), attention_output)?;
        }
        // Debug: Check x before final layer norm
        static DEBUG_BEFORE_LN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let before_ln_count = DEBUG_BEFORE_LN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if before_ln_count < 3 && x.shape()[1] >= 2 {
            if let Tensor::Cpu(x_cpu) = &x {
                let x_data = x_cpu.eval().inner().clone();
                eprintln!("DEBUG DECODER: Before final LN (first 5 dims):");
                let x0: Vec<f32> = (0..5).map(|i| x_data.get([0, 0, i])).collect();
                let x1: Vec<f32> = (0..5).map(|i| x_data.get([0, 1, i])).collect();
                eprintln!("  x[pos=0]: {:?}", x0);
                eprintln!("  x[pos=1]: {:?}", x1);
            }
        }

        let out = self.ln.forward(&x);

        // Debug: Check after final layer norm
        if before_ln_count < 3 && out.shape()[1] >= 2 {
            if let Tensor::Cpu(out_cpu) = &out {
                let out_data = out_cpu.eval().inner().clone();
                eprintln!("DEBUG DECODER: After final LN (first 5 dims):");
                let o0: Vec<f32> = (0..5).map(|i| out_data.get([0, 0, i])).collect();
                let o1: Vec<f32> = (0..5).map(|i| out_data.get([0, 1, i])).collect();
                eprintln!("  out[pos=0]: {:?}", o0);
                eprintln!("  out[pos=1]: {:?}", o1);
            }
        }

        Ok(out)
    }

    pub fn final_linear(
        &self,
        x: &Tensor<3, crate::WhisperDType>,
    ) -> Result<Tensor<3, crate::WhisperDType>> {
        let embeddings = self.token_embedding.embeddings_quantized();
        eprintln!("DEBUG DECODER: final_linear input shape: {:?}, embeddings shape: {:?}", x.shape(), embeddings.shape());

        // Debug: Compare quantized vs dequantized embedding (once)
        static DEBUG_EMBED_CMP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_EMBED_CMP.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let dequantized = self.token_embedding.embeddings();
            if let fusor::Tensor::Cpu(ref cpu_emb) = dequantized {
                let emb_data = cpu_emb.eval().inner().clone();
                let shape = dequantized.shape();

                // Sample more values from token 0
                eprintln!("DEBUG: Token 0 embedding (first 20 dims): {:?}",
                    (0..20).map(|i| emb_data.get([0, i])).collect::<Vec<_>>());

                // Compute stats for a few token embeddings
                for tok in [0, 100, 50256] {
                    let mut min_v = f32::INFINITY;
                    let mut max_v = f32::NEG_INFINITY;
                    let mut sum_sq = 0.0f32;
                    let mut non_zero = 0;
                    for i in 0..shape[1] {
                        let v = emb_data.get([tok, i]);
                        min_v = min_v.min(v);
                        max_v = max_v.max(v);
                        sum_sq += v * v;
                        if v.abs() > 1e-6 { non_zero += 1; }
                    }
                    let l2_norm = sum_sq.sqrt();
                    eprintln!("DEBUG: Token {} embedding stats: min={:.4}, max={:.4}, l2_norm={:.4}, non_zero={}/{}",
                        tok, min_v, max_v, l2_norm, non_zero, shape[1]);
                }
            }
        }

        // Debug: Print some actual values from the input to see if they vary
        static DEBUG_FINAL_LINEAR_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let count = DEBUG_FINAL_LINEAR_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count < 10 {
            match x {
                fusor::Tensor::Cpu(cpu_tensor) => {
                    let shape = x.shape();
                    eprintln!("DEBUG FINAL_LINEAR[{}]: ALL positions (first 5 dims each):", count);
                    // Get the concrete data by evaluating and getting inner
                    let concrete = cpu_tensor.eval().inner().clone();
                    for pos in 0..shape[1].min(4) {
                        let vals: Vec<f32> = (0..5.min(shape[2])).map(|i| concrete.get([0, pos, i])).collect();
                        eprintln!("  pos {}: {:?}", pos, vals);
                    }
                }
                fusor::Tensor::Gpu(_) => {
                    eprintln!("DEBUG FINAL_LINEAR[{}]: GPU tensor (not printing values)", count);
                }
            }
        }

        let logits = {
            let _enter = self.span_final.enter();
            x.q_mat_mul(embeddings)
        };
        eprintln!("DEBUG DECODER: final_linear output shape: {:?}", logits.shape());

        // Debug: check first few logit values for each position
        if count < 1 {
            let shape = logits.shape();
            eprintln!("DEBUG FINAL_LINEAR OUTPUT[{}]: shape {:?}", count, shape);
            // Get values directly by evaluating CPU tensor
            if let fusor::Tensor::Cpu(ref cpu_tensor) = logits {
                let concrete = cpu_tensor.eval();
                let inner = concrete.inner().clone();
                // Get last position's logits
                let last_pos = shape[1] - 1;

                // Verify q_mat_mul by manual computation for token 50256
                // logit[50256] should equal dot(hidden_state, embedding[50256])
                if let fusor::Tensor::Cpu(ref x_cpu) = x {
                    let x_data = x_cpu.eval().inner().clone();
                    let emb = self.token_embedding.embeddings();
                    if let fusor::Tensor::Cpu(ref emb_cpu) = emb {
                        let emb_data = emb_cpu.eval().inner().clone();
                        let tok = 50256;
                        let mut manual_dot = 0.0f32;
                        for k in 0..384 {
                            manual_dot += x_data.get([0, last_pos, k]) * emb_data.get([tok, k]);
                        }
                        let computed_logit = inner.get([0, last_pos, tok]);
                        eprintln!("  Verification for token {}:", tok);
                        eprintln!("    Manual dot product: {:.4}", manual_dot);
                        eprintln!("    Computed logit: {:.4}", computed_logit);
                        eprintln!("    Difference: {:.6}", (manual_dot - computed_logit).abs());
                    }
                }

                // Also find the max logit and its token
                let mut max_val = f32::NEG_INFINITY;
                let mut max_tok = 0;
                for i in 0..shape[2] {
                    let v = inner.get([0, last_pos, i]);
                    if v > max_val {
                        max_val = v;
                        max_tok = i;
                    }
                }
                eprintln!("  Max logit: token {} = {:.4}", max_tok, max_val);
            }
        }

        Ok(logits)
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L221
pub struct Whisper {
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
    pub config: Config,
}

impl Whisper {
    pub fn load(device: &Device, vb: &mut VarBuilder, config: Config) -> Result<Self> {
        let encoder = AudioEncoder::load(device, vb.pp("model.encoder"), &config)?;
        let decoder = TextDecoder::load(device, &mut vb.pp("model.decoder"), &config)?;
        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    pub(crate) async fn dtw_timestamps(
        attention_heads: Option<&'static [[usize; 2]]>,
        filter_width: NonZeroUsize,
        n_frames: usize,
        mask: Vec<Vec<bool>>,
        attention_output: &[TensorCache<4, crate::WhisperDType>],
    ) -> Result<Vec<Vec<crate::WhisperDType>>> {
        let Some(attention_heads) = attention_heads else {
            panic!(
                "The attention heads for word-level timestamps are not available for this model",
            );
        };

        let mut attention_output_tensor = Vec::new();
        for attn in attention_output {
            attention_output_tensor.push(attn.current_data().unwrap().clone());
        }

        extract_timestamps(
            attention_heads,
            &attention_output_tensor,
            filter_width,
            n_frames,
            mask,
        )
        .await
    }
}
