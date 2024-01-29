use crate::raw::cache::LlamaCache;
use candle_core::{Device, Tensor};
use kalosm_language_model::Session;
use std::collections::HashMap;

/// A Llama-1.5 session.
#[derive(Debug, Clone)]
pub struct LlamaSession {
    pub(crate) cache: LlamaCache,
    pub(crate) current_tokens: Vec<u32>,
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
        let tokens = self.current_tokens.clone();
        let device = self.cache.blocks[0].0.as_ref().unwrap().key.device();
        let tokens_tensor = Tensor::from_iter(tokens.iter().copied(), device).unwrap();
        let mut map = self.cache.get_tensor_map();
        map.insert("current_tokens".to_string(), tokens_tensor);
        map
    }

    /// Import a cache tensor map.
    pub fn set_tensor_map(&mut self, map: HashMap<String, Tensor>) {
        self.cache = LlamaCache::from_tensor_map(map);
    }

    /// Create a cache from a tensor map. This can be used to load a cache from disk.
    pub fn from_tensor_map(map: HashMap<String, Tensor>) -> Self {
        let current_tokens = map.get("current_tokens").unwrap().to_vec1().unwrap();
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
