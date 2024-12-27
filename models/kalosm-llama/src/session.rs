use crate::accelerated_device_if_available;
use crate::raw::cache::LlamaCache;
use candle_core::{Device, Tensor};
use kalosm_language_model::Session;
use std::collections::HashMap;

/// An error that can occur when saving or loading a [`LlamaSession`].
#[derive(Debug, thiserror::Error)]
pub enum LlamaLoadingError {
    /// An error from safetensors while loading or saving a [`LlamaSession`].
    #[error("Safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    /// An error from candle while loading or saving a [`LlamaSession`].
    #[error("Candle error: {0:?}")]
    Candle(#[from] candle_core::Error),
}

/// A Llama session with cached state for the current fed prompt
#[derive(Debug, Clone)]
pub struct LlamaSession {
    pub(crate) cache: LlamaCache,
}

impl Session for LlamaSession {
    type Error = LlamaLoadingError;

    fn write_to(&self, into: &mut Vec<u8>) -> Result<(), Self::Error> {
        let device = accelerated_device_if_available()?;
        let tensors = self.get_tensor_map(&device);
        let bytes = safetensors::serialize(&tensors, &None)?;
        into.extend_from_slice(&bytes);
        Ok(())
    }

    fn tokens(&self) -> &[u32] {
        &self.cache.tokens
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized,
    {
        let device = accelerated_device_if_available()?;
        let tensors = candle_core::safetensors::load_buffer(bytes, &device)?;

        Ok(Self::from_tensor_map(tensors)?)
    }

    fn try_clone(&self) -> Result<Self, Self::Error>
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
