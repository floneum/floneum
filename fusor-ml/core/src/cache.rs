use crate::{DataType, MaxRank, Tensor};

/// A growable tensor cache.
/// This cache manages tensor data with exponentially larger allocations as the sequence length increases.
#[derive(Debug, Clone)]
pub struct TensorCache<const R: usize, D: DataType> {
    all_data: Option<Tensor<R, D>>,
    current_seq_len: usize,
    allocated_seq_len: usize,
    concat_dim: usize,
}

impl<const R: usize, D: DataType> TensorCache<R, D> {
    /// Create a new cache with the given concatenation dimension
    pub fn new(concat_dim: usize) -> Self {
        assert!(concat_dim < R, "concat_dim must be less than tensor rank R");
        Self {
            all_data: None,
            current_seq_len: 0,
            allocated_seq_len: 0,
            concat_dim,
        }
    }

    /// Get the current data in the cache
    pub fn current_data(&self) -> Option<&Tensor<R, D>> {
        self.all_data.as_ref()
    }

    /// Reset the cache
    pub fn reset(&mut self) {
        self.all_data = None;
        self.current_seq_len = 0;
        self.allocated_seq_len = 0;
    }

    /// Append a new value to the cache
    ///
    /// Returns the full cached tensor including the newly appended data
    pub fn append(&mut self, v: &Tensor<R, D>) -> Tensor<R, D> {
        let v_shape = v.shape();
        let seq_len = v_shape[self.concat_dim];

        if let Some(cached) = &mut self.all_data {
            // First find the required new sequence length
            let required_seq_len = self.current_seq_len + seq_len;
            // Check if we need to grow the allocation
            if required_seq_len > self.allocated_seq_len {
                // Double the allocation until it's large enough
                let new_allocated_seq_len = required_seq_len.next_power_of_two();
                self.allocated_seq_len = new_allocated_seq_len;
                let new_data_shape = std::array::from_fn(|i| {
                    if i == self.concat_dim {
                        new_allocated_seq_len - self.current_seq_len
                    } else {
                        v_shape[i]
                    }
                });
                // Allocate new tensor with larger size
                let new_data = Tensor::<R, D>::zeros(v.device(), new_data_shape);
                *cached = Tensor::cat([cached.clone(), new_data], self.concat_dim);
            }
            // Assign the new data into the cached tensor
            let slice = std::array::from_fn(|i| {
                if i == self.concat_dim {
                    self.current_seq_len..required_seq_len
                } else {
                    0..v_shape[i]
                }
            });
            *cached = cached.slice_assign(slice, v);
            self.current_seq_len = required_seq_len;
            cached.clone()
        } else {
            // First append - just store it
            self.all_data = Some(v.clone());
            self.current_seq_len = seq_len;
            self.allocated_seq_len = seq_len;
            v.clone()
        }
    }
}

/// A growable KV cache for attention layers
///
/// Manages key and value caches separately, growing them as needed
#[derive(Debug, Clone)]
pub struct KvCache {
    key: TensorCache<3, f32>,
    value: TensorCache<3, f32>,
}

impl KvCache {
    /// Create a new KV cache
    ///
    /// concat_dim: The dimension along which to concatenate new key/value tensors (typically 1 for sequence length)
    pub fn new(concat_dim: usize) -> Self {
        Self {
            key: TensorCache::new(concat_dim),
            value: TensorCache::new(concat_dim),
        }
    }

    /// Get the current key data in the cache
    pub fn k(&self) -> Option<&Tensor<3, f32>> {
        self.key.current_data()
    }

    /// Get the current value data in the cache
    pub fn v(&self) -> Option<&Tensor<3, f32>> {
        self.value.current_data()
    }

    /// Reset the cache
    pub fn reset(&mut self) {
        self.key.reset();
        self.value.reset();
    }

    /// Append a new key/value pair to the cache
    ///
    /// Returns (full_keys, full_values) including the newly appended data
    pub fn append(
        &mut self,
        k: &Tensor<3, f32>,
        v: &Tensor<3, f32>,
    ) -> (Tensor<3, f32>, Tensor<3, f32>) {
        let keys = self.key.append(k);
        let values = self.value.append(v);
        (keys, values)
    }
}

/// Attention mask for causal (decoder) attention
///
/// Prevents attending to future positions
#[derive(Debug, Clone)]
pub struct AttentionMask {
    mask: Tensor<2, f32>,
}

impl AttentionMask {
    /// Create a causal mask for the given sequence length
    ///
    /// mask[i, j] = -inf if j > i (can't attend to future), 0 otherwise
    pub fn causal(device: &crate::Device, seq_len: usize) -> Self {
        // Create a lower triangular matrix of 0s and upper triangular of -inf
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }

        let mask = Tensor::new(device, mask_data.chunks(seq_len).collect::<Vec<_>>());
        Self { mask }
    }

    /// Apply the mask to attention scores
    ///
    /// attention_scores: (batch, heads, seq_len, seq_len) or similar ranks
    /// Returns: masked attention scores
    ///
    /// The mask will be broadcast to match the attention scores shape
    pub fn apply<const R: usize>(&self, attention_scores: &Tensor<R, f32>) -> Tensor<R, f32>
    where
        (Tensor<2, f32>, Tensor<R, f32>): MaxRank<R, f32>,
    {
        const {
            assert!(R >= 2, "Attention scores must have rank >= 2");
        }
        // Broadcast add
        self.mask.add_(&attention_scores)
    }

    pub fn forward<const R: usize>(&self, attention_scores: &mut Tensor<R, f32>)
    where
        (Tensor<2, f32>, Tensor<R, f32>): MaxRank<R, f32>,
    {
        *attention_scores = self.apply(attention_scores);
    }

    pub fn mask(&self) -> &Tensor<2, f32> {
        &self.mask
    }
}

/// Mask cache for efficiently managing attention masks
#[derive(Debug, Clone, Default)]
pub struct MaskCache {
    masks: std::sync::Arc<std::sync::RwLock<std::collections::HashMap<usize, AttentionMask>>>,
}

impl MaskCache {
    /// Get or create a causal mask for the given sequence length
    pub fn get_mask(
        &self,
        seq_len: usize,
        index_pos: usize,
        device: &crate::Device,
    ) -> AttentionMask {
        // Check if we have it cached
        {
            let masks = self.masks.read().unwrap();
            if let Some(mask) = masks.get(&seq_len) {
                return mask.clone();
            }
        }

        // Create and cache
        let mask = AttentionMask::causal(device, seq_len);
        {
            let mut masks = self.masks.write().unwrap();
            masks.insert(seq_len, mask.clone());
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[tokio::test]
    async fn test_tensor_cache_first_append() {
        let device = Device::new().await.unwrap();
        let mut cache = TensorCache::new(1);

        let data = [[[1.0, 2.0]], [[3.0, 4.0]]];
        let tensor = Tensor::new(&device, &data);

        let result = cache.append(&tensor);

        assert_eq!(result.shape(), &[2, 1, 2]);
        assert_eq!(cache.current_seq_len, 1);

        let output = result.as_slice().await.unwrap();
        assert_eq!(output[[0, 0, 0]], 1.0);
        assert_eq!(output[[0, 0, 1]], 2.0);
        assert_eq!(output[[1, 0, 0]], 3.0);
        assert_eq!(output[[1, 0, 1]], 4.0);
    }

    #[tokio::test]
    async fn test_tensor_cache_multiple_appends() {
        let device = Device::new().await.unwrap();
        let mut cache = TensorCache::new(1);

        let data1 = [[[1.0, 2.0]]];
        let data2 = [[[3.0, 4.0]]];
        let tensor1 = Tensor::new(&device, &data1);
        let tensor2 = Tensor::new(&device, &data2);

        cache.append(&tensor1);
        let result = cache.append(&tensor2);

        assert_eq!(result.shape(), &[1, 2, 2]);
        assert_eq!(cache.current_seq_len, 2);

        let output = result.as_slice().await.unwrap();

        println!("{output:?}");

        assert_eq!(output[[0, 0, 0]], 1.0);
        assert_eq!(output[[0, 0, 1]], 2.0);
        assert_eq!(output[[0, 1, 0]], 3.0);
        assert_eq!(output[[0, 1, 1]], 4.0);
    }

    #[tokio::test]
    async fn test_tensor_cache_reset() {
        let device = Device::new().await.unwrap();
        let mut cache = TensorCache::new(1);

        let data = [[[1.0, 2.0]]];
        let tensor = Tensor::new(&device, &data);
        cache.append(&tensor);

        cache.reset();

        assert_eq!(cache.current_seq_len, 0);
        assert!(cache.current_data().is_none());
    }

    #[tokio::test]
    async fn test_kv_cache_first_append() {
        let device = Device::new().await.unwrap();
        let mut cache = KvCache::new(1);

        let key_data = [[[1.0, 2.0]]];
        let value_data = [[[3.0, 4.0]]];
        let key = Tensor::new(&device, &key_data);
        let value = Tensor::new(&device, &value_data);

        let (k_result, v_result) = cache.append(&key, &value);

        assert_eq!(k_result.shape(), &[1, 1, 2]);
        assert_eq!(v_result.shape(), &[1, 1, 2]);

        let k_output = k_result.as_slice().await.unwrap();
        let v_output = v_result.as_slice().await.unwrap();

        assert_eq!(k_output[[0, 0, 0]], 1.0);
        assert_eq!(k_output[[0, 0, 1]], 2.0);
        assert_eq!(v_output[[0, 0, 0]], 3.0);
        assert_eq!(v_output[[0, 0, 1]], 4.0);
    }

    #[tokio::test]
    async fn test_kv_cache_multiple_appends() {
        let device = Device::new().await.unwrap();
        let mut cache = KvCache::new(1);

        let key_data1 = [[[1.0, 2.0]]];
        let value_data1 = [[[3.0, 4.0]]];
        let key1 = Tensor::new(&device, &key_data1);
        let value1 = Tensor::new(&device, &value_data1);

        cache.append(&key1, &value1);

        let key_data2 = [[[5.0, 6.0]]];
        let value_data2 = [[[7.0, 8.0]]];
        let key2 = Tensor::new(&device, &key_data2);
        let value2 = Tensor::new(&device, &value_data2);

        let (k_result, v_result) = cache.append(&key2, &value2);

        assert_eq!(k_result.shape(), &[1, 2, 2]);
        assert_eq!(v_result.shape(), &[1, 2, 2]);

        let k_output = k_result.as_slice().await.unwrap();
        let v_output = v_result.as_slice().await.unwrap();

        assert_eq!(k_output[[0, 0, 0]], 1.0);
        assert_eq!(k_output[[0, 0, 1]], 2.0);
        assert_eq!(k_output[[0, 1, 0]], 5.0);
        assert_eq!(k_output[[0, 1, 1]], 6.0);
        assert_eq!(v_output[[0, 0, 0]], 3.0);
        assert_eq!(v_output[[0, 0, 1]], 4.0);
        assert_eq!(v_output[[0, 1, 0]], 7.0);
        assert_eq!(v_output[[0, 1, 1]], 8.0);
    }

    #[tokio::test]
    async fn test_kv_cache_reset() {
        let device = Device::new().await.unwrap();
        let mut cache = KvCache::new(1);

        let key_data = [[[1.0, 2.0]]];
        let value_data = [[[3.0, 4.0]]];
        let key = Tensor::new(&device, &key_data);
        let value = Tensor::new(&device, &value_data);

        cache.append(&key, &value);

        cache.reset();

        assert!(cache.k().is_none());
        assert!(cache.v().is_none());
    }

    #[tokio::test]
    async fn test_attention_mask_causal() {
        let device = Device::new().await.unwrap();

        let seq_len = 3;
        let mask = AttentionMask::causal(&device, seq_len);

        assert_eq!(mask.mask().shape(), &[3, 3]);

        let mask_data = mask.mask().as_slice().await.unwrap();

        // Lower triangular should be 0, upper triangular should be -inf
        assert_eq!(mask_data[[0, 0]], 0.0);
        assert_eq!(mask_data[[0, 1]], f32::NEG_INFINITY);
        assert_eq!(mask_data[[0, 2]], f32::NEG_INFINITY);

        assert_eq!(mask_data[[1, 0]], 0.0);
        assert_eq!(mask_data[[1, 1]], 0.0);
        assert_eq!(mask_data[[1, 2]], f32::NEG_INFINITY);

        assert_eq!(mask_data[[2, 0]], 0.0);
        assert_eq!(mask_data[[2, 1]], 0.0);
        assert_eq!(mask_data[[2, 2]], 0.0);
    }

    #[tokio::test]
    async fn test_attention_mask_apply_4d() {
        let device = Device::new().await.unwrap();

        let mask = AttentionMask::causal(&device, 2);

        // Create attention scores: (1, 1, 2, 2)
        let scores_data = [[[[1.0, 2.0], [3.0, 4.0]]]];
        let scores = Tensor::new(&device, &scores_data);

        let masked = mask.apply(&scores);

        let output = masked.as_slice().await.unwrap();

        // [0][0] = 1.0 + 0 = 1.0
        // [0][1] = 2.0 + -inf = -inf
        // [1][0] = 3.0 + 0 = 3.0
        // [1][1] = 4.0 + 0 = 4.0
        assert_eq!(output[[0, 0, 0, 0]], 1.0);
        assert_eq!(output[[0, 0, 0, 1]], f32::NEG_INFINITY);
        assert_eq!(output[[0, 0, 1, 0]], 3.0);
        assert_eq!(output[[0, 0, 1, 1]], 4.0);
    }

    #[tokio::test]
    async fn test_attention_mask_apply_3d() {
        let device = Device::new().await.unwrap();

        let mask = AttentionMask::causal(&device, 2);

        // Create attention scores: (1, 2, 2)
        let scores_data = [[[1.0, 2.0], [3.0, 4.0]]];
        let scores = Tensor::new(&device, &scores_data);

        let masked = mask.apply(&scores);

        let output = masked.as_slice().await.unwrap();

        assert_eq!(output[[0, 0, 0]], 1.0);
        assert_eq!(output[[0, 0, 1]], f32::NEG_INFINITY);
        assert_eq!(output[[0, 1, 0]], 3.0);
        assert_eq!(output[[0, 1, 1]], 4.0);
    }

    #[tokio::test]
    async fn test_mask_cache() {
        let device = Device::new().await.unwrap();
        let cache = MaskCache::default();

        let mask1 = cache.get_mask(3, 0, &device);
        let mask2 = cache.get_mask(3, 0, &device);

        // Should be cached (same object)
        assert_eq!(mask1.mask().shape(), &[3, 3]);
        assert_eq!(mask2.mask().shape(), &[3, 3]);

        let mask3 = cache.get_mask(5, 0, &device);
        assert_eq!(mask3.mask().shape(), &[5, 5]);
    }
}
