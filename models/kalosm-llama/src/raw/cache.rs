use candle_core::{Device, Tensor};
use candle_nn::kv_cache::KvCache;
use std::collections::HashMap;

use super::LlamaConfig;

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
            blocks.push(AttentionCache(KvCache::new(2, max_seq_len)))
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
            block.0.reset()
        }
    }

    /// Get the tensor map for this cache. This can be used to save the cache to disk.
    pub fn get_tensor_map(&self, device: &Device) -> HashMap<String, Tensor> {
        let mut map = HashMap::with_capacity(self.blocks.len());
        for (i, kv_cache) in self.blocks.iter().enumerate() {
            if let (Ok(Some(k)), Ok(Some(v))) = (kv_cache.0.k(), kv_cache.0.v()) {
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
                    blocks.resize(i + 1, AttentionCache(KvCache::new(2, max_seq_len)));
                }
                if k.ends_with(".key") {
                    match blocks.get_mut(i) {
                        Some(AttentionCache(kv_cache)) => {
                            let key = kv_cache.k_cache_mut();
                            key.reset();
                            key.append(&v)?;
                        }
                        _ => {
                            let mut kv_cache = KvCache::new(2, max_seq_len);
                            kv_cache.k_cache_mut().append(&v)?;
                            blocks[i] = AttentionCache(kv_cache);
                        }
                    }
                } else if k.ends_with(".value") {
                    match blocks.get_mut(i) {
                        Some(AttentionCache(kv_cache)) => {
                            let value = kv_cache.v_cache_mut();
                            value.reset();
                            value.append(&v)?;
                        }
                        _ => {
                            let mut kv_cache = KvCache::new(2, max_seq_len);
                            kv_cache.k_cache_mut().append(&v)?;
                            blocks[i] = AttentionCache(kv_cache)
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

#[derive(Debug, Clone)]
pub(crate) struct AttentionCache(pub(crate) KvCache);
