use fusor_core::layers::RmsNorm;
use fusor_core::{Device, Result, Tensor, VarBuilder};

use super::layer::QwenLayer;
use super::rope::RopeCache;
use crate::raw::embedding::{embedding, Embedding};

/// Configuration for QwenEmbeddingModel loaded from GGUF metadata
#[derive(Debug, Clone)]
pub struct QwenConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub head_dimension: usize,
    pub context_length: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

impl QwenConfig {
    /// Load configuration from GGUF metadata
    pub fn from_gguf(vb: &VarBuilder) -> Result<Self> {
        let num_heads = vb
            .get_metadata(".attention.head_count")
            .ok()
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(16) as usize;

        let num_kv_heads = vb
            .get_metadata(".attention.head_count_kv")
            .ok()
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(num_heads as u32) as usize;

        let num_layers = vb
            .get_metadata(".block_count")
            .ok()
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(28) as usize;

        let hidden_size = vb
            .get_metadata(".embedding_length")
            .ok()
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(1024) as usize;

        let context_length = vb
            .get_metadata(".context_length")
            .ok()
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(32768) as usize;

        let rope_theta = vb
            .get_metadata(".rope.freq_base")
            .ok()
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1_000_000.0);

        let rms_norm_eps = vb
            .get_metadata(".attention.layer_norm_rms_epsilon")
            .ok()
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1e-6);

        // Use attention.key_length for head dimension (like kalosm-llama)
        // Fall back to hidden_size / num_heads if not present
        let head_dimension = vb
            .get_metadata(".attention.key_length")
            .ok()
            .and_then(|v| v.to_u32().ok())
            .map(|x| x as usize)
            .unwrap_or_else(|| hidden_size / num_heads);

        Ok(Self {
            num_heads,
            num_kv_heads,
            num_layers,
            hidden_size,
            head_dimension,
            context_length,
            rope_theta,
            rms_norm_eps,
        })
    }
}

/// Qwen embedding model (encoder-only for embeddings)
pub struct QwenEmbeddingModel {
    token_embeddings: Embedding,
    layers: Vec<QwenLayer>,
    final_norm: RmsNorm<1, f32>,
    rope_cache: RopeCache,
    pub(crate) device: Device,
    config: QwenConfig,
}

impl QwenEmbeddingModel {
    /// Load QwenEmbeddingModel from GGUF weights
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let config = QwenConfig::from_gguf(vb)?;

        // Load token embeddings
        let token_embeddings = embedding(device, &mut vb.pp("token_embd"))?;

        // Create RoPE cache
        let rope_cache = RopeCache::new(&config, device)?;

        // Load transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer = QwenLayer::load(
                device,
                &mut vb.pp(format!("blk.{i}")),
                config.num_heads,
                config.num_kv_heads,
                config.head_dimension,
                config.rms_norm_eps,
            )?;
            layers.push(layer);
        }

        // Load final layer norm
        let final_norm = RmsNorm::load(device, &mut vb.pp("output_norm"), config.rms_norm_eps)?;

        Ok(Self {
            token_embeddings,
            layers,
            final_norm,
            rope_cache,
            device: device.clone(),
            config,
        })
    }

    /// Forward pass through the model
    ///
    /// Returns: [batch_size, seq_len, hidden_size]
    pub fn forward(
        &self,
        input_ids: &Tensor<2, u32>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3, f32> {
        // Get token embeddings
        let mut hidden_states = self.token_embeddings.forward(input_ids);

        // Pass through transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &self.rope_cache, 0, attention_mask);
        }

        // Apply final layer norm
        self.final_norm.forward(&hidden_states)
    }

    /// Get the maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.config.context_length
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.config.hidden_size
    }
}
