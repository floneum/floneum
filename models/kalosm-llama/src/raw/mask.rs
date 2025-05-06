use fusor_core::{Device, Tensor};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

#[derive(Default)]
pub struct MaskCache {
    masks: RwLock<HashMap<usize, AttentionMask>>,
}

impl MaskCache {
    pub fn get_mask(&self, seq_len: usize, seqlen_offset: usize, device: &Device) -> AttentionMask {
        let mask = if let Some(mask) = {
            let masks = self.masks.read().unwrap();
            masks.get(&seq_len).cloned()
        } {
            mask
        } else {
            let mask = (0..seq_len)
                .map(|i| {
                    (0..seq_len)
                        .map(move |j| u32::from(j > i))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mask = Tensor::new(device, &mask);
            let mut masks = self.masks.write().unwrap();
            let mask = AttentionMask {
                mask,
                on_true: OnceLock::new(),
            };
            masks.insert(seq_len, mask.clone());
            mask
        };

        let mask_tensor = if seqlen_offset > 0 {
            // If this isn't the first token, we need to pad the mask with zeros for the previous tokens.
            let mask0 = Tensor::splat(device, 0, [seq_len, seqlen_offset]);
            let last_dim = mask.mask.shape().len() - 1;
            Tensor::cat([mask0, mask.mask], last_dim)
        } else {
            mask.mask
        };

        AttentionMask {
            mask: mask_tensor,
            on_true: mask.on_true,
        }
    }
}

#[derive(Clone)]
pub struct AttentionMask {
    pub mask: Tensor<2, u32>,
    pub on_true: OnceLock<Tensor<3, f32>>,
}

impl AttentionMask {
    pub fn forward(&self, attn_weights: &mut Tensor<3, f32>) {
        let shape = attn_weights.shape();
        let attention_mask = self.mask.broadcast(*shape);
        let on_true = self
            .on_true
            .get_or_init(|| Tensor::splat(attn_weights.device(), -3.4028235e35, *shape));
        *attn_weights = attention_mask.where_cond(on_true.clone(), attn_weights.clone());
    }
}
