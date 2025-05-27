use candle_core::Tensor;

/// A growable kv cache. This cache wraps candles [`KvCache`] with exponentially larger allocations as the sequence length increases.
#[derive(Debug, Clone)]
pub struct KvCache {
    key: TensorCache,
    value: TensorCache,
}

impl KvCache {
    /// Create a new cache with the given max sequence length.
    pub fn new(concat_dim: usize, max_seq_len: usize) -> Self {
        Self {
            key: TensorCache::new(concat_dim, max_seq_len),
            value: TensorCache::new(concat_dim, max_seq_len),
        }
    }

    /// Get the current key data in the cache.
    pub fn k(&self) -> candle_core::Result<Option<Tensor>> {
        self.key.current_data()
    }

    /// Get the key cache.
    pub fn k_cache(&self) -> &TensorCache {
        &self.key
    }

    /// Get the key cache mutably.
    pub fn k_cache_mut(&mut self) -> &mut TensorCache {
        &mut self.key
    }

    /// Get the current value data in the cache.
    pub fn v(&self) -> candle_core::Result<Option<Tensor>> {
        self.value.current_data()
    }

    /// Get the value cache.
    pub fn v_cache(&self) -> &TensorCache {
        &self.value
    }

    /// Get the value cache mutably.
    pub fn v_cache_mut(&mut self) -> &mut TensorCache {
        &mut self.value
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.key.reset();
        self.value.reset();
    }

    /// Append a new key/value pair to the cache.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        self.key.append(k)?;
        self.value.append(v)?;
        Ok((
            self.key.current_data()?.unwrap(),
            self.value.current_data()?.unwrap(),
        ))
    }
}

/// A growable tensor cache. This cache wraps candles [`Cache`] with exponentially larger allocations as the sequence length increases.
#[derive(Debug, Clone)]
pub struct TensorCache {
    all_data: Option<Tensor>,
    start_offset: usize,
    current_seq_len: usize,
    allocated_seq_len: usize,
    concat_dim: usize,
    max_seq_len: usize,
}

impl TensorCache {
    /// Create a new cache with the given max sequence length.
    pub fn new(concat_dim: usize, max_seq_len: usize) -> Self {
        Self {
            all_data: None,
            start_offset: 0,
            current_seq_len: 0,
            allocated_seq_len: 0,
            concat_dim,
            max_seq_len,
        }
    }

    /// Get the all tensor data in the cache.
    pub fn all_data(&self) -> &Option<Tensor> {
        &self.all_data
    }

    /// Get the current data in the cache.
    pub fn current_data(&self) -> candle_core::Result<Option<Tensor>> {
        let data = match self.all_data.as_ref() {
            None => None,
            Some(d) => Some(d.narrow(self.concat_dim, self.start_offset, self.current_seq_len)?.contiguous()?),
        };
        Ok(data)
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.all_data = None;
        self.current_seq_len = 0;
        self.allocated_seq_len = 0;
        self.start_offset = 0;
    }

    /// Append a new value to the cache.
    pub fn append(&mut self, v: &Tensor) -> candle_core::Result<()> {
        let v = v.contiguous()?;
        let seq_len = v.dim(self.concat_dim)?;

        let current_allocated_size = self.allocated_seq_len;
        let size_required_for_append = self.current_seq_len + seq_len;

        // If the required size is larger than the max sequence length, cut the start of the cache.
        if size_required_for_append > self.max_seq_len {
            const EXTRA: usize = 0;
            let max_seq_len_without_extra = self.max_seq_len - EXTRA;
            let new_start = size_required_for_append - max_seq_len_without_extra;
            // Cut the start of the cache.
            let all_data = self.all_data.as_ref().unwrap().narrow(
                self.concat_dim,
                new_start,
                current_allocated_size - new_start,
            )?;
            let mut shape = v.shape().dims().to_vec();
            shape[self.concat_dim] = EXTRA;
            let empty = Tensor::zeros(shape.as_slice(), v.dtype(), v.device())?;
            let all_data = Tensor::cat(&[all_data, v.clone(), empty], self.concat_dim)?;
            assert_eq!(
                all_data.dim(self.concat_dim)?,
                max_seq_len_without_extra + EXTRA
            );
            self.all_data = Some(all_data);
            self.current_seq_len = max_seq_len_without_extra;
            self.allocated_seq_len = self.max_seq_len;
        } else {
            // If adding the new key/value pair would exceed the max sequence length, we need to allocate a new tensor with double the size or the max sequence length whichever is smaller.
            if size_required_for_append > current_allocated_size {
                // The new size of the cache is double the old size or the max sequence length of the model.
                // We try to keep the new size a power of two to keep memory alignment nice.
                let next_power_of_two = size_required_for_append.next_power_of_two();
                let new_cache_max_seq_len = next_power_of_two.min(self.max_seq_len);
                tracing::trace!(
                    "Extending Tensor cache from {current_allocated_size} to {new_cache_max_seq_len}"
                );

                // Create a new tensor with the new size.
                let mut tensors = Vec::new();
                if let Some(v) = self.all_data() {
                    tensors.push(v.clone());
                }
                // Append a new blank tensor with the remaining size.
                let mut shape = v.shape().dims().to_vec();
                shape[self.concat_dim] = new_cache_max_seq_len - current_allocated_size;
                tensors.push(Tensor::zeros(shape.as_slice(), v.dtype(), v.device())?);
                let new_cache = Tensor::cat(&tensors, self.concat_dim)?;
                // Replace the old cache with the new cache.
                self.all_data = Some(new_cache);
                self.allocated_seq_len = new_cache_max_seq_len;
            }

            self.all_data
                .as_mut()
                .unwrap()
                .slice_set(&v, self.concat_dim, self.current_seq_len)?;
            self.current_seq_len += seq_len;
        }

        Ok(())
    }
}
