use candle_core::Tensor;

/// A growable kv cache. This cache wraps candles [`KvCache`] with exponentially larger allocations as the sequence length increases.
#[derive(Debug, Clone)]
pub struct KvCache {
    cache: candle_nn::kv_cache::KvCache,
    concat_dim: usize,
    max_seq_len: usize,
}

impl KvCache {
    /// Create a new cache with the given max sequence length.
    pub fn new(concat_dim: usize, max_seq_len: usize) -> Self {
        Self {
            cache: candle_nn::kv_cache::KvCache::new(concat_dim, 8),
            concat_dim,
            max_seq_len,
        }
    }

    /// Get the raw cache.
    pub fn cache(&self) -> &candle_nn::kv_cache::KvCache {
        &self.cache
    }

    /// Get the raw cache mutably.
    pub fn cache_mut(&mut self) -> &mut candle_nn::kv_cache::KvCache {
        &mut self.cache
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.cache.reset()
    }

    /// Append a new key/value pair to the cache.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let seq_len = k.dim(self.concat_dim)?;
        // The key and value token length must be the same.
        debug_assert_eq!(seq_len, v.dim(self.concat_dim)?);

        let current_allocated_size = self.cache.k_cache().max_seq_len();
        let size_required_for_append = self.cache.current_seq_len() + seq_len;

        // If adding the new key/value pair would exceed the max sequence length, we need to allocate a new tensor with double the size or the max sequence length whichever is smaller.
        if size_required_for_append > current_allocated_size {
            // The new size of the cache is double the old size or the max sequence length of the model.
            // We try to keep the new size a power of two to keep memory alignment nice.
            let next_power_of_two = size_required_for_append.next_power_of_two();
            let new_cache_max_seq_len = next_power_of_two.min(self.max_seq_len);

            // Create a new cache with the new size.
            let mut new_cache =
                candle_nn::kv_cache::KvCache::new(self.concat_dim, new_cache_max_seq_len);
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

/// A growable tensor cache. This cache wraps candles [`Cache`] with exponentially larger allocations as the sequence length increases.
#[derive(Debug, Clone)]
pub struct TensorCache {
    cache: candle_nn::kv_cache::Cache,
    concat_dim: usize,
    max_seq_len: usize,
}

impl TensorCache {
    /// Create a new cache with the given max sequence length.
    pub fn new(concat_dim: usize, max_seq_len: usize) -> Self {
        Self {
            cache: candle_nn::kv_cache::Cache::new(concat_dim, 8),
            concat_dim,
            max_seq_len,
        }
    }

    /// Get the raw cache.
    pub fn cache(&self) -> &candle_nn::kv_cache::Cache {
        &self.cache
    }

    /// Get the raw cache mutably.
    pub fn cache_mut(&mut self) -> &mut candle_nn::kv_cache::Cache {
        &mut self.cache
    }

    /// Get the current tensor in the cache.
    pub fn all_data(&self) -> &Option<Tensor> {
        self.cache.all_data()
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.cache.reset()
    }

    /// Append a new value to the cache.
    pub fn append(&mut self, v: &Tensor) -> candle_core::Result<()> {
        let v = v.contiguous()?;
        let seq_len = v.dim(self.concat_dim)?;
        // The key and value token length must be the same.
        debug_assert_eq!(seq_len, v.dim(self.concat_dim)?);

        let current_allocated_size = self.cache.max_seq_len();
        let size_required_for_append = self.cache.current_seq_len() + seq_len;

        // If adding the new key/value pair would exceed the max sequence length, we need to allocate a new tensor with double the size or the max sequence length whichever is smaller.
        if size_required_for_append > current_allocated_size {
            // The new size of the cache is double the old size or the max sequence length of the model.
            // We try to keep the new size a power of two to keep memory alignment nice.
            let next_power_of_two = size_required_for_append.next_power_of_two();
            let new_cache_max_seq_len = next_power_of_two.min(self.max_seq_len);

            // Create a new cache with the new size.
            let mut new_cache =
                candle_nn::kv_cache::Cache::new(self.concat_dim, new_cache_max_seq_len);
            // Append the old cache to the new cache.
            if let Some(v) = self.cache.all_data() {
                new_cache.append(&v.contiguous()?)?;
            }
            // Replace the old cache with the new cache.
            self.cache = new_cache;
        }

        // self.cache.append(&k, &v)
        self.cache.append(&v)
    }
}
