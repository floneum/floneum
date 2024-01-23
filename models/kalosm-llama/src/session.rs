use crate::raw::Model;
use candle_core::{Device, Tensor};
use kalosm_language_model::Session;
use std::collections::HashMap;

/// A Llama-1.5 session.
#[derive(Debug, Clone)]
pub struct LlamaSession {
    pub(crate) cache: LlamaCache,
}

impl Session for LlamaSession {
    fn save_to(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let tensors = self.get_tensor_map();
        Ok(candle_core::safetensors::save(&tensors, path)?)
    }

    fn load_from(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self>
    where
        Self: std::marker::Sized,
    {
        let device = Device::cuda_if_available(0)?;
        let tensors = candle_core::safetensors::load(path, &device)?;

        Ok(Self::from_tensor_map(tensors))
    }

    fn try_clone(&self) -> anyhow::Result<Self>
    where
        Self: std::marker::Sized,
    {
        Ok(self.clone())
    }
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
    pub fn from_tensor_map(map: HashMap<String, Tensor>) -> Self {
        Self {
            cache: LlamaCache::from_tensor_map(map),
        }
    }
}

/// A cache for Llama inference. This cache will speed up generation of sequential text significantly.
#[derive(Debug, Clone)]
pub struct LlamaCache {
    pub(crate) tokens: Vec<u32>,
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
            tokens: Vec::new(),
            blocks,
        }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        for block in &mut self.blocks {
            *block = AttentionCache(None)
        }
        self.tokens.clear();
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
