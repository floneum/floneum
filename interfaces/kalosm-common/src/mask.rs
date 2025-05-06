use candle_core::*;
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

#[derive(Default, Debug)]
pub struct MaskCache {
    masks: RwLock<HashMap<(usize, Option<usize>), AttentionMask>>,
}

impl MaskCache {
    pub fn get_mask(
        &self,
        seq_len: usize,
        seqlen_offset: usize,
        sliding_window_size: Option<usize>,
        device: &Device,
    ) -> Result<AttentionMask> {
        let mask = if let Some(mask) = {
            let masks = self.masks.read().unwrap();
            masks.get(&(seq_len, sliding_window_size)).cloned()
        } {
            mask
        } else {
            let mask: Vec<_> = if let Some(sliding_window_size) = sliding_window_size {
                (0..seq_len)
                    .flat_map(|i| {
                        (0..seq_len).map(move |j| u8::from(i < j || j + sliding_window_size <= i))
                    })
                    .collect()
            } else {
                (0..seq_len)
                    .flat_map(|i| (0..seq_len).map(move |j| u8::from(i < j)))
                    .collect()
            };
            let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
            let mut masks = self.masks.write().unwrap();
            let mask = AttentionMask {
                mask,
                on_true: OnceLock::new(),
            };
            masks.insert((seq_len, sliding_window_size), mask.clone());
            mask
        };

        let mask_tensor = if seqlen_offset > 0 {
            // If this isn't the first token, we need to pad the mask with zeros for the previous tokens.
            let mask0 = Tensor::zeros((seq_len, seqlen_offset), DType::U8, device)?;
            Tensor::cat(&[&mask0, &mask.mask], D::Minus1)?
        } else {
            mask.mask
        };

        let mask_tensor = mask_tensor.unsqueeze(0)?.unsqueeze(0)?;

        Ok(AttentionMask {
            mask: mask_tensor,
            on_true: mask.on_true,
        })
    }
}

#[test]
fn test_sliding_window() {
    let device = Device::Cpu;
    let mask_cache = MaskCache::default();
    let mask = mask_cache.get_mask(4, 0, Some(2), &device).unwrap().mask;
    let mask = mask.squeeze(0).unwrap().squeeze(0).unwrap();
    assert_eq!(mask.shape().dims(), &[4, 4]);
    assert_eq!(
        mask.to_vec2::<u8>().unwrap(),
        vec![
            vec![0, 1, 1, 1],
            vec![0, 0, 1, 1],
            vec![1, 0, 0, 1],
            vec![1, 1, 0, 0],
        ]
    );

    let mask = mask_cache.get_mask(4, 0, None, &device).unwrap().mask;
    let mask = mask.squeeze(0).unwrap().squeeze(0).unwrap();
    assert_eq!(mask.shape().dims(), &[4, 4]);
    assert_eq!(
        mask.to_vec2::<u8>().unwrap(),
        vec![
            vec![0, 1, 1, 1],
            vec![0, 0, 1, 1],
            vec![0, 0, 0, 1],
            vec![0, 0, 0, 0],
        ]
    );
}

#[derive(Clone, Debug)]
pub struct AttentionMask {
    mask: Tensor,
    on_true: OnceLock<Tensor>,
}

impl AttentionMask {
    pub fn forward(&self, attn_weights: &mut Tensor) -> candle_core::Result<()> {
        let shape = attn_weights.shape();
        let attention_mask = self.mask.broadcast_as(shape)?;
        let on_true = match self.on_true.get() {
            Some(on_true) => on_true.clone(),
            None => {
                let on_true =
                    Tensor::new(f32::NEG_INFINITY, attn_weights.device())?.broadcast_as(shape)?;
                self.on_true.set(on_true.clone()).unwrap();
                on_true
            }
        };
        *attn_weights = attention_mask.where_cond(&on_true, attn_weights)?;
        Ok(())
    }
}
