use candle_core::{Device, Tensor};
use std::collections::HashMap;

/// A cache for Llama inference. This cache will speed up generation of sequential text significantly.
#[derive(Debug, Clone)]
pub struct LlamaCache {
    pub(crate) tokens: Vec<u32>,
    pub(crate) blocks: Vec<AttentionCache>,
}

impl LlamaCache {
    /// Create a new cache for a model
    pub fn new(layers: usize) -> Self {
        let mut blocks = Vec::with_capacity(layers);
        for _ in 0..layers {
            blocks.push(AttentionCache(None))
        }
        Self {
            tokens: Vec::new(),
            blocks,
        }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        for block in &mut self.blocks {
            *block = AttentionCache(None)
        }
    }

    /// Get the tensor map for this cache. This can be used to save the cache to disk.
    pub fn get_tensor_map(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            if let AttentionCache(Some(AttentionCacheValue { key, value })) = block {
                map.insert(format!("Llama.cache.blocks.{}.key", i), key.clone());
                map.insert(format!("Llama.cache.blocks.{}.value", i), value.clone());
            }
        }
        map.insert(
            "Llama.cache.tokens".to_string(),
            Tensor::from_iter(self.tokens.iter().copied(), &Device::Cpu).unwrap(),
        );
        map
    }

    /// Create a cache from a tensor map. This can be used to load a cache from disk.
    pub fn from_tensor_map(map: HashMap<String, Tensor>) -> Self {
        let tokens = map
            .get("Llama.cache.tokens")
            .and_then(|tokens| tokens.to_vec1().ok())
            .unwrap_or_default();
        let mut blocks = Vec::with_capacity(24);
        for (k, v) in map {
            if let Some(i) = k.strip_prefix("Llama.cache.blocks.") {
                let i = i
                    .strip_suffix(".key")
                    .unwrap_or_else(|| i.strip_suffix(".value").unwrap());
                let i = i.parse::<usize>().unwrap_or(0);
                if i >= blocks.len() {
                    blocks.resize(i + 1, AttentionCache(None));
                }
                if k.ends_with(".key") {
                    match blocks.get_mut(i) {
                        Some(AttentionCache(Some(AttentionCacheValue { key, value: _ }))) => {
                            *key = v;
                        }
                        _ => {
                            blocks[i] = AttentionCache(Some(AttentionCacheValue {
                                key: v.clone(),
                                value: v,
                            }));
                        }
                    }
                } else if k.ends_with(".value") {
                    match blocks.get_mut(i) {
                        Some(AttentionCache(Some(AttentionCacheValue { key: _, value }))) => {
                            *value = v;
                        }
                        _ => {
                            blocks[i] = AttentionCache(Some(AttentionCacheValue {
                                key: v.clone(),
                                value: v,
                            }));
                        }
                    }
                }
            }
        }
        Self { tokens, blocks }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AttentionCache(pub(crate) Option<AttentionCacheValue>);

#[derive(Debug, Clone)]
pub(crate) struct AttentionCacheValue {
    pub(crate) key: Tensor,
    pub(crate) value: Tensor,
}
