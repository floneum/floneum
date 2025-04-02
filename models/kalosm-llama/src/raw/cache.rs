use fusor_core::{DataType, Device, Tensor};
use std::collections::HashMap;

use super::LlamaConfig;

/// The dimension along which the attention cache is concatenated with attention for new tokens.
const CONCAT_DIMENSION: usize = 1;

/// A cache for llama inference. This cache will speed up generation of sequential text significantly.
#[derive(Clone)]
pub(crate) struct LlamaCache {
    pub(crate) tokens: Vec<u32>,
    pub(crate) blocks: Vec<KvCache>,
}

impl LlamaCache {
    /// Create a new cache for a model
    pub fn new(config: &LlamaConfig) -> Self {
        let max_seq_len = config.context_length;
        let mut blocks = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            blocks.push(KvCache::new(CONCAT_DIMENSION, max_seq_len))
        }
        Self {
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

    // /// Get the tensor map for this cache. This can be used to save the cache to disk.
    // pub fn get_tensor_map(&self, device: &Device) -> HashMap<String, Tensor> {
    //     let mut map = HashMap::with_capacity(self.blocks.len());
    //     for (i, kv_cache) in self.blocks.iter().enumerate() {
    //         if let (Ok(Some(k)), Ok(Some(v))) = (kv_cache.cache().k(), kv_cache.cache().v()) {
    //             map.insert(
    //                 format!("llama.cache.blocks.{}.key", i),
    //                 k.to_device(device).unwrap(),
    //             );
    //             map.insert(
    //                 format!("llama.cache.blocks.{}.value", i),
    //                 v.to_device(device).unwrap(),
    //             );
    //         }
    //     }
    //     if !self.tokens.is_empty() {
    //         // Tensor from iter panics or segfaults if the iterator is empty
    //         map.insert(
    //             "llama.cache.tokens".to_string(),
    //             Tensor::from_iter(self.tokens.iter().copied(), device).unwrap(),
    //         );
    //     }
    //     map.insert(
    //         "llama.cache.max_seq_len".to_string(),
    //         Tensor::new(self.max_seq_len as u32, device).unwrap(),
    //     );
    //     map
    // }

    // /// Create a cache from a tensor map. This can be used to load a cache from disk.
    // pub fn from_tensor_map(map: HashMap<String, Tensor>) -> fusor_core::Result<Self> {
    //     let tokens: Vec<u32> = map
    //         .get("llama.cache.tokens")
    //         .and_then(|tokens| tokens.to_vec1().ok())
    //         .unwrap_or_default();
    //     let max_seq_len = map
    //         .get("llama.cache.max_seq_len")
    //         .and_then(|max_seq_len| max_seq_len.to_scalar::<u32>().ok())
    //         .unwrap_or(2048) as usize;
    //     let mut blocks = Vec::with_capacity(24);
    //     for (k, v) in map {
    //         if let Some(i) = k.strip_prefix("llama.cache.blocks.") {
    //             let i = i
    //                 .strip_suffix(".key")
    //                 .unwrap_or_else(|| i.strip_suffix(".value").unwrap());
    //             let i = i.parse::<usize>().unwrap_or(0);
    //             if i >= blocks.len() {
    //                 blocks.resize(i + 1, KvCache::new(CONCAT_DIMENSION, max_seq_len));
    //             }
    //             if k.ends_with(".key") {
    //                 match blocks.get_mut(i) {
    //                     Some(cache) => {
    //                         let key_cache = cache.cache_mut().k_cache_mut();
    //                         let len = v.dim(CONCAT_DIMENSION)?;
    //                         *key_cache = Cache::new(CONCAT_DIMENSION, len);
    //                         key_cache.append(&v)?;
    //                     }
    //                     _ => {
    //                         let mut cache = KvCache::new(CONCAT_DIMENSION, max_seq_len);
    //                         let key_cache = cache.cache_mut().k_cache_mut();
    //                         let len = v.dim(CONCAT_DIMENSION)?;
    //                         *key_cache = Cache::new(CONCAT_DIMENSION, len);
    //                         key_cache.append(&v)?;
    //                         blocks[i] = cache;
    //                     }
    //                 }
    //             } else if k.ends_with(".value") {
    //                 match blocks.get_mut(i) {
    //                     Some(cache) => {
    //                         let value_cache = cache.cache_mut().v_cache_mut();
    //                         let len = v.dim(CONCAT_DIMENSION)?;
    //                         *value_cache = Cache::new(CONCAT_DIMENSION, len);
    //                         value_cache.append(&v)?;
    //                     }
    //                     _ => {
    //                         let mut cache = KvCache::new(CONCAT_DIMENSION, max_seq_len);
    //                         let value_cache = cache.cache_mut().v_cache_mut();
    //                         let len = v.dim(CONCAT_DIMENSION)?;
    //                         *value_cache = Cache::new(CONCAT_DIMENSION, len);
    //                         value_cache.append(&v)?;
    //                         blocks[i] = cache;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     Ok(Self {
    //         tokens,
    //         blocks,
    //         max_seq_len,
    //     })
    // }
}

#[derive(Clone)]
pub(crate) struct KvCache {
    k_cache: TensorCache<3, f32>,
    v_cache: TensorCache<3, f32>,
}

impl KvCache {
    /// Create a new cache for a model
    pub fn new(concat_dim: usize, max_seq_len: usize) -> Self {
        Self {
            k_cache: TensorCache::new(concat_dim, max_seq_len),
            v_cache: TensorCache::new(concat_dim, max_seq_len),
        }
    }

    /// Get the key cache.
    pub fn k_cache(&self) -> &TensorCache<3, f32> {
        &self.k_cache
    }

    /// Get the value cache.
    pub fn v_cache(&self) -> &TensorCache<3, f32> {
        &self.v_cache
    }

    /// Get the key cache mutable reference.
    pub fn k_cache_mut(&mut self) -> &mut TensorCache<3, f32> {
        &mut self.k_cache
    }

    /// Get the value cache mutable reference.
    pub fn v_cache_mut(&mut self) -> &mut TensorCache<3, f32> {
        &mut self.v_cache
    }

    /// Append a new key/value pair to the cache.
    pub fn append(
        &mut self,
        k: &Tensor<3, f32>,
        v: &Tensor<3, f32>,
    ) -> (Tensor<3, f32>, Tensor<3, f32>) {
        // The key and value token length must be the same.
        debug_assert_eq!(k.shape()[CONCAT_DIMENSION], v.shape()[CONCAT_DIMENSION]);
        self.k_cache.append(k);
        self.v_cache.append(v);
        (
            self.k_cache.all_data().unwrap(),
            self.v_cache.all_data().unwrap(),
        )
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.k_cache.reset();
        self.v_cache.reset();
    }
}

/// A growable tensor cache. This cache wraps candles [`Cache`] with exponentially larger allocations as the sequence length increases.
#[derive(Clone)]
pub struct TensorCache<const R: usize, D> {
    cache: Option<Tensor<R, D>>,
    concat_dim: usize,
    filled_size: usize,
    max_size: usize,
}

impl<const R: usize, D: DataType> TensorCache<R, D> {
    /// Create a new cache with the given max sequence length.
    pub fn new(concat_dim: usize, max_seq_len: usize) -> Self {
        assert!(
            concat_dim < R,
            "concat_dim must be less than the number of dimensions"
        );
        assert!(max_seq_len > 0, "max_seq_len must be greater than 0");
        Self {
            cache: None,
            concat_dim,
            filled_size: 0,
            max_size: max_seq_len,
        }
    }

    /// Get the current tensor in the cache.
    pub fn all_data(&self) -> Option<Tensor<R, D>> {
        self.cache.as_ref().map(|t| {
            t.slice(std::array::from_fn(|i| {
                if i == self.concat_dim {
                    0..self.filled_size
                } else {
                    0..t.shape()[i]
                }
            }))
        })
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.filled_size = 0;
    }

    /// Get the current allocated size of the cache.
    fn allocated_size(&self) -> usize {
        self.cache
            .as_ref()
            .map(|t| t.shape()[self.concat_dim])
            .unwrap_or(0)
    }

    /// Append a new value to the cache.
    pub fn append(&mut self, v: &Tensor<R, D>) {
        let seq_len = v.shape()[self.concat_dim];
        // The key and value token length must be the same.
        debug_assert_eq!(seq_len, v.shape()[self.concat_dim]);

        let current_allocated_size = self.allocated_size();
        let size_required_for_append = self.filled_size + seq_len;

        // If adding the new key/value pair would exceed the max sequence length, we need to allocate a new tensor with double the size or the max sequence length whichever is smaller.
        if size_required_for_append > current_allocated_size {
            // The new size of the cache is double the old size or the max sequence length of the model.
            // We try to keep the new size a power of two to keep memory alignment nice.
            let next_power_of_two = size_required_for_append.next_power_of_two();
            let new_cache_max_seq_len = next_power_of_two.min(self.max_size);
            tracing::trace!(
                "Extending Tensor cache from {current_allocated_size} to {new_cache_max_seq_len}"
            );

            // Create a new cache with the new size.
            let new_shape = std::array::from_fn(|i| {
                if i == self.concat_dim {
                    new_cache_max_seq_len
                } else {
                    v.shape()[i]
                }
            });
            let cache = match self.cache.take() {
                Some(cache) => {
                    let mut cache = cache.resize(new_shape);
                    cache = cache.slice_assign(
                        std::array::from_fn(|i| {
                            if i == self.concat_dim {
                                self.filled_size..self.filled_size + seq_len
                            } else {
                                0..v.shape()[i]
                            }
                        }),
                        v,
                    );
                    cache
                }
                None => {
                    // If the cache is empty, we need to create a new one.
                    v.resize(new_shape)
                }
            };
            // Replace the old cache with the new cache.
            self.cache = Some(cache);
        } else {
            let mut cache = self.cache.take().unwrap();
            cache = cache.slice_assign(
                std::array::from_fn(|i| {
                    if i == self.concat_dim {
                        self.filled_size..self.filled_size + seq_len
                    } else {
                        0..v.shape()[i]
                    }
                }),
                v,
            );
            self.cache = Some(cache);
        }
    }
}
