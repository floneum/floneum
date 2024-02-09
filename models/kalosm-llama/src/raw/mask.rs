use candle_core::*;
use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Default)]
pub struct MaskCache {
    masks: RwLock<HashMap<usize, Tensor>>,
}

impl MaskCache {
    pub fn get_mask(&self, seq_len: usize, seqlen_offset: usize) -> Result<Tensor> {
        let mask = if let Some(mask) = {
            let masks = self.masks.read().unwrap();
            masks.get(&seq_len).cloned()
        } {
            mask
        } else {
            let mask: Vec<_> = (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (seq_len, seq_len), &Device::Cpu)?;
            let mut masks = self.masks.write().unwrap();
            masks.insert(seq_len, mask.clone());
            mask
        };

        let mask = if seqlen_offset > 0 {
            // If this isn't the first token, we need to pad the mask with zeros for the previous tokens.
            let mask0 = Tensor::zeros((seq_len, seqlen_offset), DType::U8, &Device::Cpu)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };

        let mask = mask.unsqueeze(0)?.unsqueeze(0)?;

        Ok(mask)
    }
}
