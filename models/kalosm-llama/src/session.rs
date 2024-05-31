use crate::accelerated_device_if_available;
use crate::raw::cache::LlamaCache;
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
        let device = accelerated_device_if_available()?;
        let tensors = self.get_tensor_map(&device);
        Ok(candle_core::safetensors::save(&tensors, path)?)
    }

    fn load_from(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self>
    where
        Self: std::marker::Sized,
    {
        let device = accelerated_device_if_available()?;
        let tensors = candle_core::safetensors::load(path, &device)?;

        Ok(Self::from_tensor_map(tensors)?)
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
    pub fn get_tensor_map(&self, device: &Device) -> HashMap<String, Tensor> {
        self.cache.get_tensor_map(device)
    }

    /// Import a cache tensor map.
    pub fn set_tensor_map(&mut self, map: HashMap<String, Tensor>) -> candle_core::Result<()> {
        self.cache = LlamaCache::from_tensor_map(map)?;
        Ok(())
    }

    /// Create a cache from a tensor map. This can be used to load a cache from disk.
    pub fn from_tensor_map(map: HashMap<String, Tensor>) -> candle_core::Result<Self> {
        Ok(Self {
            cache: LlamaCache::from_tensor_map(map)?,
        })
    }
}
