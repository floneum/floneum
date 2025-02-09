use candle_core::*;
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

#[derive(Default, Debug)]
pub struct MaskCache {
    masks: RwLock<HashMap<usize, AttentionMask>>,
}

impl MaskCache {
    pub fn get_mask(
        &self,
        seq_len: usize,
        seqlen_offset: usize,
        device: &Device,
    ) -> Result<AttentionMask> {
        let mask = if let Some(mask) = {
            let masks = self.masks.read().unwrap();
            masks.get(&seq_len).cloned()
        } {
            mask
        } else {
            let mask: Vec<_> = (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
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

#[derive(Clone, Debug)]
pub struct AttentionMask {
    pub mask: Tensor,
    pub on_true: OnceLock<Tensor>,
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
