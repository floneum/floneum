use candle_core::{Device, Tensor};
use candle_nn::kv_cache::{Cache, KvCache};
use std::collections::HashMap;

use super::LlamaConfig;

/// The dimension along which the attention cache is concatenated with attention for new tokens.
const CONCAT_DIMENSION: usize = 2;

/// A cache for llama inference. This cache will speed up generation of sequential text significantly.
#[derive(Debug, Clone)]
pub struct LlamaCache {
    max_seq_len: usize,
    pub(crate) tokens: Vec<u32>,
    pub(crate) blocks: Vec<AttentionCache>,
}

impl LlamaCache {
    /// Create a new cache for a model
    pub fn new(config: &LlamaConfig) -> Self {
        let max_seq_len = config.context_length;
        let mut blocks = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            blocks.push(AttentionCache::new(max_seq_len))
        }
        Self {
            max_seq_len,
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

    /// Get the tensor map for this cache. This can be used to save the cache to disk.
    pub fn get_tensor_map(&self, device: &Device) -> HashMap<String, Tensor> {
        let mut map = HashMap::with_capacity(self.blocks.len());
        for (i, kv_cache) in self.blocks.iter().enumerate() {
            if let (Ok(Some(k)), Ok(Some(v))) = (kv_cache.cache.k(), kv_cache.cache.v()) {
                map.insert(format!("llama.cache.blocks.{}.key", i), k);
                map.insert(format!("llama.cache.blocks.{}.value", i), v);
            }
        }
        map.insert(
            "llama.cache.tokens".to_string(),
            Tensor::from_iter(self.tokens.iter().copied(), device).unwrap(),
        );
        map.insert(
            "llama.cache.max_seq_len".to_string(),
            Tensor::new(self.max_seq_len as u32, device).unwrap(),
        );
        map
    }

    /// Create a cache from a tensor map. This can be used to load a cache from disk.
    pub fn from_tensor_map(map: HashMap<String, Tensor>) -> candle_core::Result<Self> {
        let tokens = map
            .get("llama.cache.tokens")
            .and_then(|tokens| tokens.to_vec1().ok())
            .unwrap_or_default();
        let max_seq_len = map
            .get("llama.cache.max_seq_len")
            .and_then(|max_seq_len| max_seq_len.to_scalar::<u32>().ok())
            .unwrap_or(2048) as usize;
        let mut blocks = Vec::with_capacity(24);
        for (k, v) in map {
            if let Some(i) = k.strip_prefix("llama.cache.blocks.") {
                let i = i
                    .strip_suffix(".key")
                    .unwrap_or_else(|| i.strip_suffix(".value").unwrap());
                let i = i.parse::<usize>().unwrap_or(0);
                if i >= blocks.len() {
                    blocks.resize(i + 1, AttentionCache::new(max_seq_len));
                }
                if k.ends_with(".key") {
                    match blocks.get_mut(i) {
                        Some(AttentionCache { cache, .. }) => {
                            let key_cache = cache.k_cache_mut();
                            let len = v.dim(CONCAT_DIMENSION)?;
                            *key_cache = Cache::new(CONCAT_DIMENSION, len);
                            key_cache.append(&v)?;
                        }
                        _ => {
                            let mut cache = AttentionCache::new(max_seq_len);
                            let key_cache = cache.cache.k_cache_mut();
                            let len = v.dim(CONCAT_DIMENSION)?;
                            *key_cache = Cache::new(CONCAT_DIMENSION, len);
                            key_cache.append(&v)?;
                            blocks[i] = cache;
                        }
                    }
                } else if k.ends_with(".value") {
                    match blocks.get_mut(i) {
                        Some(AttentionCache { cache, .. }) => {
                            let value_cache = cache.v_cache_mut();
                            let len = v.dim(CONCAT_DIMENSION)?;
                            *value_cache = Cache::new(CONCAT_DIMENSION, len);
                            value_cache.append(&v)?;
                        }
                        _ => {
                            let mut cache = AttentionCache::new(max_seq_len);
                            let value_cache = cache.cache.v_cache_mut();
                            let len = v.dim(CONCAT_DIMENSION)?;
                            *value_cache = Cache::new(CONCAT_DIMENSION, len);
                            value_cache.append(&v)?;
                            blocks[i] = cache;
                        }
                    }
                }
            }
        }
        Ok(Self {
            tokens,
            blocks,
            max_seq_len,
        })
    }
}

/// A cache for the attention layer. This cache wraps candles [`KvCache`] with exponentially larger allocations as the sequence length increases.
#[derive(Debug, Clone)]
pub(crate) struct AttentionCache {
    cache: KvCache,
    max_seq_len: usize,
}

impl AttentionCache {
    /// Create a new cache with the given max sequence length.
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            cache: KvCache::new(CONCAT_DIMENSION, 8),
            max_seq_len,
        }
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.cache.reset()
    }

    /// Append a new key/value pair to the cache.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let seq_len = k.dim(CONCAT_DIMENSION)?;
        // The key and value token length must be the same.
        debug_assert_eq!(seq_len, v.dim(CONCAT_DIMENSION)?);

        let current_allocated_size = self.cache.k_cache().max_seq_len();
        let size_required_for_append = self.cache.current_seq_len() + seq_len;

        // If adding the new key/value pair would exceed the max sequence length, we need to allocate a new tensor with double the size or the max sequence length whichever is smaller.
        if size_required_for_append > current_allocated_size {
            // The new size of the cache is double the old size or the max sequence length of the model.
            // We try to keep the new size a power of two to keep memory alignment nice.
            let next_power_of_two = size_required_for_append.next_power_of_two();
            let new_cache_max_seq_len = next_power_of_two.min(self.max_seq_len);

            // Create a new cache with the new size.
            let mut new_cache = KvCache::new(CONCAT_DIMENSION, new_cache_max_seq_len);
            // Append the old cache to the new cache.
            if let (Ok(Some(k)), Ok(Some(v))) = (self.cache.k(), self.cache.v()) {
                new_cache.k_cache_mut().append(&k.contiguous()?)?;
                new_cache.v_cache_mut().append(&v.contiguous()?)?;
            }
            // Replace the old cache with the new cache.
            self.cache = new_cache;
        }

        self.cache.append(&k, &v)
    }
}
