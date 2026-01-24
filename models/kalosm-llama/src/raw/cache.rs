use fusor::cache::KvCache;
use fusor::DataType;

use super::LlamaConfig;

/// The dimension along which the attention cache is concatenated with attention for new tokens.
const CONCAT_DIMENSION: usize = 2;

/// A cache for llama inference. This cache will speed up generation of sequential text significantly.
#[derive(Clone)]
pub struct LlamaCache<F: DataType = f32> {
    pub(crate) start_time: u32,
    pub(crate) tokens: Vec<u32>,
    pub(crate) blocks: Vec<KvCache<F>>,
}

impl<F: DataType> LlamaCache<F> {
    /// Create a new cache for a model
    pub fn new<G: fusor::FloatDataType>(config: &LlamaConfig<G>) -> Self {
        let max_seq_len = config.context_length;
        let mut blocks = Vec::with_capacity(config.n_layer);
        for layer_idx in 0..config.n_layer {
            let max_seq_len = if let (Some(sliding_window_type), Some(sliding_window_size)) =
                (config.sliding_window_type, config.sliding_window_size)
            {
                let is_sliding = (layer_idx + 1) % sliding_window_type != 0;
                if is_sliding {
                    sliding_window_size
                } else {
                    max_seq_len
                }
            } else {
                max_seq_len
            };
            blocks.push(KvCache::new(CONCAT_DIMENSION, max_seq_len))
        }
        Self {
            start_time: 0,
            tokens: Vec::new(),
            blocks,
        }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        for block in &mut self.blocks {
            block.reset()
        }
    }
}
