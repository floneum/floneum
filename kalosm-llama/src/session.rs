use candle_core::Tensor;
pub use candle_transformers::quantized_var_builder::VarBuilder;
use std::collections::HashMap;
use crate::raw::Model;

/// A Llama-1.5 session.
pub struct LlamaSession {
    pub(crate) cache: LlamaCache,
    pub(crate) current_tokens: Vec<u32>,
}

impl LlamaSession {
    /// Export the current cache tensor map.
    pub fn get_tensor_map(&self) -> HashMap<String, Tensor> {
        self.cache.get_tensor_map()
    }

    /// Import a cache tensor map.
    pub fn set_tensor_map(&mut self, map: HashMap<String, Tensor>) {
        self.cache = LlamaCache::from_tensor_map(map);
    }

    /// Create a cache from a tensor map. This can be used to load a cache from disk.
    pub fn from_tensor_map(map: HashMap<String, Tensor>, current_tokens: Vec<u32>) -> Self {
        Self {
            cache: LlamaCache::from_tensor_map(map),
            current_tokens,
        }
    }

    /// Get the current tokens.
    pub fn get_current_tokens(&self) -> &[u32] {
        &self.current_tokens
    }
}

/// A cache for Llama inference. This cache will speed up generation of sequential text significantly.
#[derive(Debug, Clone)]
pub struct LlamaCache {
    pub(crate) blocks: Vec<AttentionCache>,
}

impl LlamaCache {
    /// Create a new cache for a model
    pub fn new(model: &Model) -> Self {
        let mut blocks = Vec::with_capacity(model.layers.len());
        for _ in 0..model.layers.len() {
            blocks.push(AttentionCache(None))
        }
        Self {
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
        map
    }

    /// Create a cache from a tensor map. This can be used to load a cache from disk.
    pub fn from_tensor_map(map: HashMap<String, Tensor>) -> Self {
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
        Self {
            blocks,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AttentionCache(pub(crate) Option<AttentionCacheValue>);

#[derive(Debug, Clone)]
pub(crate) struct AttentionCacheValue {
    pub(crate) key: Tensor,
    pub(crate) value: Tensor,
}
