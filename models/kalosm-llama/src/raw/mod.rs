use std::path::PathBuf;
use std::sync::Arc;

use crate::chat_template::HuggingFaceChatTemplate;
use crate::raw::attention_layer::LlamaAttention;
use crate::LlamaSourceError;
use attention_layer::AttentionBias;
use attention_layer::AttentionVariant;
use attention_layer::FeedForwardVariant;
use attention_layer::GroupedAttention;
use attention_layer::LlamaFeedForward;
use attention_layer::PhiFeedForward;
use attention_layer::SeparateAttention;
use candle_core::quantized::gguf_file::Value;
use candle_core::quantized::*;
use candle_core::IndexOp;
use candle_core::Module;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Embedding;
use candle_transformers::quantized_nn::Linear;
use candle_transformers::quantized_nn::RmsNorm;
use kalosm_common::qmatmul_from_qtensor;
use kalosm_common::MaskCache;

mod attention_layer;
pub mod cache;
mod rope;
mod silu;
mod vision;

use cache::LlamaCache;
use kalosm_language_model::MediaHints;
use rope::RopeImplementation;

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
    pub(crate) vision_start_token: Option<u32>,
    pub(crate) _vision_end_token: Option<u32>,
    pub(crate) image_pad_token: Option<u32>,
    pub(crate) video_pad_token: Option<u32>,
    pub(crate) mrope_sections: Option<Vec<usize>>,
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
            vision_start_token: None,
            _vision_end_token: None,
            image_pad_token: None,
            video_pad_token: None,
            mrope_sections: None,
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
    vision_encoder: Option<vision::QwenVisionTransformer>,
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
            vision_start_token: None,
            _vision_end_token: None,
            image_pad_token: None,
            video_pad_token: None,
            mrope_sections: None,
        };
        let config = Arc::new(config);
        let rope = RopeImplementation::new(&config, DType::F32, config.rope_theta, device)?;
        let tok_embeddings_q = ct.remove("tok_embeddings.weight")?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let output = if let Ok(output) = ct.remove("output.weight") {
            qmatmul_from_qtensor(output)?
        } else {
            // If there is no output layer, assume the word embeddings are tied to the output
            qmatmul_from_qtensor(tok_embeddings_q)?
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
                attention_wq: qmatmul_from_qtensor(attention_wq)?,
                attention_q_norm: None,
                attention_wk: qmatmul_from_qtensor(attention_wk)?,
                attention_k_norm: None,
                attention_wv: qmatmul_from_qtensor(attention_wv)?,
                interleaved_rope: true,
                bias: None,
            });
            let feed_forward_variant = FeedForwardVariant::Llama(LlamaFeedForward::new(
                qmatmul_from_qtensor(feed_forward_w1)?,
                qmatmul_from_qtensor(feed_forward_w2)?,
                qmatmul_from_qtensor(feed_forward_w3)?,
            ));
            layers.push(LlamaAttention {
                attention_variant,
                attention_wo: Linear::from_arc(attention_wo.into(), None)?,
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
            vision_encoder: None,
            masks: Default::default(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        source: &mut ShardedGguf<R>,
        vision_ct: Option<gguf_file::Content>,
        vision_file: Option<PathBuf>,
        device: &Device,
        override_stop_token_string: Option<String>,
        override_chat_template: Option<String>,
        rope_scaling: Option<RopeScalingConfig>,
    ) -> std::result::Result<Self, LlamaSourceError> {
        // Get the eos and bos tokens from the metadata
        let tokens: std::result::Result<Vec<_>, _> = source
            .get("tokenizer.ggml.tokens")?
            .to_vec()?
            .iter()
            .map(|v| v.to_string().cloned())
            .collect();
        let tokens = tokens?;
        let start_token = source
            .get("tokenizer.ggml.bos_token_id")
            .ok()
            .and_then(|v| v.to_u32().ok());
        let stop_token = if let Some(override_stop_token_string) = override_stop_token_string {
            tokens
                .iter()
                .position(|v| **v == override_stop_token_string)
                .unwrap_or(0) as u32
        } else {
            source.get("tokenizer.ggml.eos_token_id")?.to_u32()?
        };
        let start_token_string = start_token
            .map(|v| tokens[v as usize].clone())
            .unwrap_or_default();
        let stop_token_string = tokens[stop_token as usize].clone();
        let chat_template = override_chat_template.or_else(|| {
            source
                .get("tokenizer.chat_template")
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
        let architecture = source.get("general.architecture")?.to_string()?.clone();
        let head_count = source.get(".attention.head_count")?.to_u32()? as usize;
        let head_count_kv = source.get(".attention.head_count_kv")?.to_u32()? as usize;
        let block_count = source.get(".block_count")?.to_u32()? as usize;
        let embedding_length = source.get(".embedding_length")?.to_u32()? as usize;
        // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
        let rms_norm_eps = source.get(".attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let rope_freq_base = source
            .get(".rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);
        let sliding_window_size = source
            .get(".attention.sliding_window")
            .and_then(|m| m.to_u32())
            .ok()
            .map(|x| x as usize);
        let sliding_window_type = source
            .get(".attention.sliding_window_type")
            .and_then(|m| m.to_u32())
            .ok()
            .map(|x| x as usize)
            .or_else(|| (architecture == "gemma3").then_some(GEMMA_DEFAULT_SLIDING_WINDOW_TYPE));

        let rope_freq_base_sliding = source
            .get(".rope.local_freq_base")
            .and_then(|m| m.to_f32())
            .ok()
            .or_else(|| (architecture == "gemma3").then_some(GEMMA_DEFAULT_ROPE_FREQUENCY_SLIDING));

        let context_length = source.get(".context_length")?.to_u32()? as usize;
        let head_dim = source
            .get(".attention.key_length")
            .and_then(|v| v.to_u32())
            .ok()
            .map(|x| x as usize)
            .unwrap_or_else(|| embedding_length / head_count);

        let config = LlamaConfig {
            rope_freq_weight: match source.tensor("rope_freqs.weight", device).ok() {
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
            vision_start_token: tokens
                .iter()
                .position(|v| *v == "<|vision_start|>")
                .map(|v| v as u32),
            _vision_end_token: tokens
                .iter()
                .position(|v| *v == "<|vision_end|>")
                .map(|v| v as u32),
            image_pad_token: tokens
                .iter()
                .position(|v| *v == "<|image_pad|>")
                .map(|v| v as u32),
            video_pad_token: tokens
                .iter()
                .position(|v| *v == "<|video_pad|>")
                .map(|v| v as u32),
            mrope_sections: source
                .get(".rope.dimension_sections")
                .ok()
                .and_then(|m| {
                    m.to_vec()
                        .ok()
                        .map(|v| v.iter().map(|x| x.to_i32().map(|x| x as usize)).collect())
                })
                .transpose()?,
        };
        let config = Arc::new(config);

        let rope = RopeImplementation::new(&config, DType::F32, config.rope_theta, device)?;
        let sliding_rope = rope_freq_base_sliding
            .map(|rope_freq_base_sliding| {
                RopeImplementation::new(&config, DType::F32, rope_freq_base_sliding, device)
            })
            .transpose()?;

        let tok_embeddings_q = source.tensor("token_embd.weight", device)?;
        let mut tok_embeddings = tok_embeddings_q.dequantize(device)?;
        // if this is gemma3, scale the tok_embeddings by sqrt(embedding_length)
        if architecture == "gemma3" {
            tok_embeddings = (tok_embeddings * (embedding_length as f64).sqrt())?;
        }
        let tok_embeddings = Embedding::new(tok_embeddings, embedding_length);

        let norm = source.tensor("output_norm.weight", device)?;
        let norm = decode_norm(norm, rms_norm_eps)?;
        let output = if let Ok(output) = source.tensor("output.weight", device) {
            qmatmul_from_qtensor(output)?
        } else {
            // If there is no output layer, assume the word embeddings are tied to the output
            qmatmul_from_qtensor(tok_embeddings_q)?
        };
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_variant =
                if let Ok(qkv) = source.tensor(&format!("{prefix}.attn_qkv.weight"), device) {
                    AttentionVariant::Grouped(GroupedAttention {
                        attention_qkv: qmatmul_from_qtensor(qkv)?,
                    })
                } else {
                    let q = source.tensor(&format!("{prefix}.attn_q.weight"), device)?;
                    let k = source.tensor(&format!("{prefix}.attn_k.weight"), device)?;
                    let v = source.tensor(&format!("{prefix}.attn_v.weight"), device)?;
                    let bias = if let (Ok(bias_q), Ok(bias_k), Ok(bias_v)) = (
                        source.tensor(&format!("{prefix}.attn_q.bias"), device),
                        source.tensor(&format!("{prefix}.attn_k.bias"), device),
                        source.tensor(&format!("{prefix}.attn_v.bias"), device),
                    ) {
                        Some(AttentionBias::from_qtensor(&bias_q, &bias_k, &bias_v)?)
                    } else {
                        None
                    };
                    let q_norm = source
                        .tensor(&format!("{prefix}.attn_q_norm.weight"), device)
                        .ok();
                    let k_norm = source
                        .tensor(&format!("{prefix}.attn_k_norm.weight"), device)
                        .ok();
                    let separate = SeparateAttention {
                        attention_wq: qmatmul_from_qtensor(q)?,
                        attention_q_norm: q_norm
                            .map(|norm| decode_norm(norm, rms_norm_eps))
                            .transpose()?,
                        attention_wk: qmatmul_from_qtensor(k)?,
                        attention_k_norm: k_norm
                            .map(|norm| decode_norm(norm, rms_norm_eps))
                            .transpose()?,
                        attention_wv: qmatmul_from_qtensor(v)?,
                        interleaved_rope: architecture != "qwen2" && architecture != "gemma3",
                        bias,
                    };
                    AttentionVariant::Separate(separate)
                };
            let attention_wo = source.tensor(&format!("{prefix}.attn_output.weight"), device)?;
            // Try to read from the up, down and gate weights
            let feed_forward_variant = if let Ok(ffn_gate) =
                source.tensor(&format!("{prefix}.ffn_gate.weight"), device)
            {
                let feed_forward_w1 = ffn_gate;
                let feed_forward_w2 =
                    source.tensor(&format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 = source.tensor(&format!("{prefix}.ffn_up.weight"), device)?;
                FeedForwardVariant::Llama(LlamaFeedForward::new(
                    qmatmul_from_qtensor(feed_forward_w1)?,
                    qmatmul_from_qtensor(feed_forward_w2)?,
                    qmatmul_from_qtensor(feed_forward_w3)?,
                ))
            } else {
                // Otherwise, try to read from the up, and down weights
                let up = source.tensor(&format!("{prefix}.ffn_up.weight"), device)?;
                // Transpose the down tensor
                let down = source.tensor(&format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_length = source.get(".feed_forward_length")?.to_u32()? as usize;

                FeedForwardVariant::Phi(PhiFeedForward {
                    up: qmatmul_from_qtensor(up)?,
                    down: qmatmul_from_qtensor(down)?,
                    feed_forward_length,
                })
            };
            let attention_norm = source.tensor(&format!("{prefix}.attn_norm.weight"), device)?;
            let post_attention_norm = source
                .tensor(&format!("{prefix}.post_attention_norm.weight"), device)
                .ok();
            let ffn_norm = source.tensor(&format!("{prefix}.ffn_norm.weight"), device)?;
            let ffn_post_norm = source
                .tensor(&format!("{prefix}.post_ffw_norm.weight"), device)
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
                attention_wo: Linear::from_arc(attention_wo.into(), None)?,
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

        // If the model is a vision model, load the vision encoder
        let vision_encoder: Option<std::result::Result<vision::QwenVisionTransformer, _>> =
            if let (Some(vision_ct), Some(vision_file)) = (vision_ct, vision_file) {
                Some(vision::QwenVisionTransformer::from_gguf(
                    vision_ct,
                    &vision_file,
                    device,
                ))
            } else {
                None
            };

        Ok(Self {
            config,
            tok_embeddings,
            layers,
            norm,
            output,
            masks: Default::default(),
            vision_encoder: vision_encoder.transpose()?,
        })
    }

    pub fn encode_tokens(
        &self,
        raw_tokens: &[u32],
        raw_images: &[(image::DynamicImage, MediaHints)],
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
    ) -> Result<(Tensor, usize, usize, Option<Tensor>)> {
        let mut grid_thw = Vec::new();
        let mut images = Vec::new();
        let mut image_token_ranges = Vec::new();
        // Embed all images
        if let Some(vision_encoder) = &self.vision_encoder {
            for (image, hints) in raw_images {
                let min_pixels = hints.min_tokens();
                let max_pixels = hints.max_tokens();
                let (image, thw) =
                    vision_encoder.preprocess_image(image, min_pixels, max_pixels)?;
                images.push(image);
                grid_thw.push(thw)
            }
        }

        // Add any image padding tokens to the tokens if needed
        let tokens = if let (Some(image_pad_token), Some(vision_start_token), Some(vision)) = (
            self.config.image_pad_token,
            self.config.vision_start_token,
            &self.vision_encoder,
        ) {
            let mut tokens = Vec::new();
            let mut token_iter = raw_tokens.iter().copied();
            let mut image_iter = grid_thw.iter();
            while let Some(token) = token_iter.next() {
                tokens.push(token);
                let start_index = tokens.len();
                if token == vision_start_token {
                    match token_iter.next() {
                        Some(next) if next == image_pad_token => {
                            // Push a pad token for every image token
                            let grid = image_iter.next().ok_or_else(|| {
                                candle_core::Error::Msg(
                                    "Image pad token found without matching image.".to_string(),
                                )
                            })?;
                            for _ in 0..grid.iter().product::<u32>()
                                / (vision.spacial_merge_size as u32).pow(2)
                            {
                                tokens.push(image_pad_token);
                            }
                            image_token_ranges.push(start_index..tokens.len());
                        }
                        Some(next) => {
                            tokens.push(next);
                        }
                        None => break,
                    }
                }
            }
            tokens
        } else {
            raw_tokens.to_vec()
        };

        let mut seq_len = tokens.len();
        let cached_tokens = cache.as_ref().map(|c| c.tokens.len()).unwrap_or_default();
        // We use a lower cutoff than the context length to avoid recomputing the attention every single token
        let cutoff_len: usize = self.config.context_length.saturating_sub(32).max(8);
        let (tokens, index_pos, start_time) = if seq_len + cached_tokens
            > self.config.context_length
        {
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
            (all_tokens.to_vec(), 0, 0)
        } else {
            let index_pos = cache.as_ref().map(|c| c.tokens.len()).unwrap_or_default();
            let start_time = cache.as_ref().map(|c| c.start_time).unwrap_or_default();
            if let Some(cache) = cache.as_mut() {
                cache.tokens.extend_from_slice(&tokens);
            }
            (tokens, index_pos, start_time)
        };
        let x = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

        let mut embeddings = self.tok_embeddings.forward(&x)?;
        let mut pos_ids = None;
        let batch_size = embeddings.dim(0)?;
        let embed_dim = embeddings.dim(2)?;

        if let Some(vision_encoder) = &self.vision_encoder {
            for ((pixels, grid), range) in images.iter().zip(&grid_thw).zip(image_token_ranges) {
                let image_embeds = vision_encoder.forward_image(pixels, *grid)?;
                embeddings = embeddings.slice_assign(
                    &[0..batch_size, range, 0..embed_dim],
                    &image_embeds.unsqueeze(0)?,
                )?;
            }
            let (new_pos_ids, new_start_time) =
                vision_encoder.get_rope_index(&tokens, &grid_thw, &self.config, start_time)?;
            if let Some(cache) = cache.as_mut() {
                cache.start_time = new_start_time;
            }
            pos_ids = Some(new_pos_ids);
        }

        Ok((embeddings, seq_len, index_pos, pos_ids))
    }

    pub fn forward(
        &self,
        tokens: &[u32],
        images: &[(image::DynamicImage, MediaHints)],
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
    ) -> Result<Tensor> {
        let (mut layer_in, seq_len, index_pos, pos_ids) =
            self.encode_tokens(tokens, images, device, cache.as_deref_mut())?;

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
                pos_ids.as_ref(),
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

fn debug_assert_none_nan(#[allow(unused)] tensor: &Tensor) {
    #[cfg(feature = "extra_assertions")]
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

pub(crate) struct ShardedGguf<R: std::io::Read + std::io::Seek> {
    contents: Vec<(gguf_file::Content, R)>,
}

impl<R: std::io::Read + std::io::Seek> ShardedGguf<R> {
    pub fn new(contents: Vec<(gguf_file::Content, R)>) -> Self {
        Self { contents }
    }

    pub fn get(&self, name: &str) -> Result<&Value> {
        if name.starts_with('.') {
            if let Some(value) = self
                .contents
                .iter()
                .flat_map(|(k, _)| k.metadata.iter().filter(|(k, _)| k.ends_with(name)))
                .min_by_key(|(k, _)| k.len())
                .map(|(_, v)| v)
            {
                return Ok(value);
            }
        } else {
            for (content, _) in &self.contents {
                if let Some(value) = content.metadata.get(name) {
                    return Ok(value);
                }
            }
        }
        candle_core::bail!("cannot find {name} in metadata")
    }

    pub fn tensor(&mut self, name: &str, device: &Device) -> Result<QTensor> {
        for (content, r) in &mut self.contents {
            if let Ok(value) = content.tensor(r, name, device) {
                return Ok(value);
            }
        }
        candle_core::bail!("cannot find {name} in tensors")
    }
}
