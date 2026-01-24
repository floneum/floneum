//! Rotary Position Embeddings (RoPE) that work on both CPU and GPU backends.

use crate::{AddOp, ConcreteTensor, Tensor, MulOp, NegOp, SimdBinaryOp, SimdElement, SimdUnaryOp, SubOp};
use fusor_core::{DataType, FloatDataType};
use fusor_cpu::FloatOps;

fn rotate_half<D>(
    xs: &Tensor<4, D, ConcreteTensor<D, 4>>,
) -> Tensor<4, D, ConcreteTensor<D, 4>>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default + std::ops::Neg<Output = D>,
    NegOp: SimdUnaryOp<D>,
{
    let shape = xs.shape();
    let last_dim = shape[3];
    let xs1 = xs.narrow(3, 0, last_dim / 2);
    let xs2 = xs.narrow(3, last_dim / 2, last_dim - last_dim / 2);
    let neg_xs2 = match xs2 {
        Tensor::Cpu(t) => Tensor::Cpu((-t).to_concrete()),
        Tensor::Gpu(t) => Tensor::Gpu(-t),
    };
    crate::cat([neg_xs2, xs1], 3)
}

impl<D> Tensor<4, D, ConcreteTensor<D, 4>>
where
    D: SimdElement
        + DataType
        + FloatDataType
        + FloatOps
        + Default
        + std::ops::Add<Output = D>
        + std::ops::Sub<Output = D>
        + std::ops::Mul<Output = D>
        + std::ops::Neg<Output = D>,
    AddOp: SimdBinaryOp<D>,
    SubOp: SimdBinaryOp<D>,
    MulOp: SimdBinaryOp<D>,
    NegOp: SimdUnaryOp<D>,
{
    /// Apply rotary position embedding (normal mode).
    ///
    /// This pairs first half with second half: (0, head_dim/2), (1, head_dim/2+1), etc.
    ///
    /// # Arguments
    /// * `cos` - Cosine positional embeddings, shape (seq_len, head_dim/2)
    /// * `sin` - Sine positional embeddings, shape (seq_len, head_dim/2)
    pub fn rope(
        &self,
        cos: &Tensor<2, D, ConcreteTensor<D, 2>>,
        sin: &Tensor<2, D, ConcreteTensor<D, 2>>,
    ) -> Self {
        let [_, _, sequence_length, _] = self.shape();

        let cos = crate::cat([cos.clone(), cos.clone()], 1);
        let sin = crate::cat([sin.clone(), sin.clone()], 1);

        let cos = cos.narrow(0, 0, sequence_length);
        let sin = sin.narrow(0, 0, sequence_length);

        let cos: Tensor<4, D, _> = cos.unsqueeze(0).unsqueeze(0);
        let sin: Tensor<4, D, _> = sin.unsqueeze(0).unsqueeze(0);

        let rotated = rotate_half(self);
        self.dispatch_quad(
            &cos,
            &sin,
            &rotated,
            |s, c, sn, r| {
                // Use broadcasting mul_ and add_
                let sc = s.as_ref().mul_(c.as_ref());
                let rsn = r.as_ref().mul_(sn.as_ref());
                sc.add_(rsn)
            },
            |s, c, sn, r| s.mul_(c) + r.mul_(sn),
        )
    }

    /// Apply interleaved rotary position embedding.
    ///
    /// This pairs adjacent elements: (0, 1), (2, 3), etc.
    ///
    /// # Arguments
    /// * `cos` - Cosine positional embeddings, shape (seq_len, head_dim/2)
    /// * `sin` - Sine positional embeddings, shape (seq_len, head_dim/2)
    pub fn rope_interleaved(
        &self,
        cos: &Tensor<2, D, ConcreteTensor<D, 2>>,
        sin: &Tensor<2, D, ConcreteTensor<D, 2>>,
    ) -> Self {
        let [bz, n_head, sequence_length, embed] = self.shape();

        let cos: Tensor<5, D, _> = cos
            .narrow(0, 0, sequence_length)
            .reshape([sequence_length, embed / 2, 1])
            .broadcast_as([bz, 1, sequence_length, embed / 2, 1]);
        let sin: Tensor<5, D, _> = sin
            .narrow(0, 0, sequence_length)
            .reshape([sequence_length, embed / 2, 1])
            .broadcast_as([bz, 1, sequence_length, embed / 2, 1]);
        let x: Tensor<5, D, _> = self.reshape([bz, n_head, sequence_length, embed / 2, 2]);

        let x0 = x.narrow(4, 0, 1);
        let x1 = x.narrow(4, 1, 1);

        let y0 = x0.dispatch_quad(
            &cos,
            &x1,
            &sin,
            |a, c, b, s| {
                // Use broadcasting mul_ and sub_
                let ac = a.as_ref().mul_(c.as_ref());
                let bs = b.as_ref().mul_(s.as_ref());
                ac.sub_(bs)
            },
            |a, c, b, s| &a.mul_(c) - &b.mul_(s),
        );
        let y1 = x0.dispatch_quad(
            &sin,
            &x1,
            &cos,
            |a, s, b, c| {
                // Use broadcasting mul_ and add_
                let as_ = a.as_ref().mul_(s.as_ref());
                let bc = b.as_ref().mul_(c.as_ref());
                as_.add_(bc)
            },
            |a, s, b, c| &a.mul_(s) + &b.mul_(c),
        );

        crate::cat([y0, y1], 4).flatten_last_n::<1, 4>()
    }

    /// Apply fused interleaved RoPE (rotary position embedding).
    /// This pairs adjacent elements: (0, 1), (2, 3), etc.
    ///
    /// On GPU, this uses an optimized fused kernel. On CPU, it delegates to `rope_interleaved`.
    pub fn rope_fused(
        &self,
        cos: &Tensor<2, D, ConcreteTensor<D, 2>>,
        sin: &Tensor<2, D, ConcreteTensor<D, 2>>,
    ) -> Self {
        match (self, cos, sin) {
            // GPU path - use the optimized fused kernel
            (Tensor::Gpu(x), Tensor::Gpu(cos), Tensor::Gpu(sin)) => {
                Tensor::Gpu(x.rope_fused(cos, sin))
            }
            // CPU path - use composite operations
            (Tensor::Cpu(_), Tensor::Cpu(_), Tensor::Cpu(_)) => {
                self.rope_interleaved(cos, sin)
            }
            _ => panic!("All tensors must be on the same device"),
        }
    }

    /// Apply fused normal RoPE (rotary position embedding).
    /// This pairs first half with second half: (0, head_dim/2), (1, head_dim/2+1), etc.
    ///
    /// On GPU, this uses an optimized fused kernel. On CPU, it delegates to `rope`.
    pub fn rope_normal_fused(
        &self,
        cos: &Tensor<2, D, ConcreteTensor<D, 2>>,
        sin: &Tensor<2, D, ConcreteTensor<D, 2>>,
    ) -> Self {
        match (self, cos, sin) {
            // GPU path - use the optimized fused kernel
            (Tensor::Gpu(x), Tensor::Gpu(cos), Tensor::Gpu(sin)) => {
                Tensor::Gpu(x.rope_normal_fused(cos, sin))
            }
            // CPU path - use composite operations
            (Tensor::Cpu(_), Tensor::Cpu(_), Tensor::Cpu(_)) => {
                self.rope(cos, sin)
            }
            _ => panic!("All tensors must be on the same device"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rope_cpu() {
        let pos_shape = [11, 32];
        let cos_data: Vec<Vec<f32>> = (0..pos_shape[0])
            .map(|i| {
                (0..pos_shape[1])
                    .map(|j| {
                        ((i as f32) / 10000f32.powf((2 * j) as f32 / (pos_shape[1] * 2) as f32))
                            .cos()
                    })
                    .collect()
            })
            .collect();
        let sin_data: Vec<Vec<f32>> = (0..pos_shape[0])
            .map(|i| {
                (0..pos_shape[1])
                    .map(|j| {
                        ((i as f32) / 10000f32.powf((2 * j) as f32 / (pos_shape[1] * 2) as f32))
                            .sin()
                    })
                    .collect()
            })
            .collect();

        let cos_flat: Vec<f32> = cos_data.iter().flatten().copied().collect();
        let sin_flat: Vec<f32> = sin_data.iter().flatten().copied().collect();
        let cos: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [pos_shape[0], pos_shape[1]],
            &cos_flat,
        ));
        let sin: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [pos_shape[0], pos_shape[1]],
            &sin_flat,
        ));

        // Input: [1, 3, 11, 64]
        let shape = [1, 3, 11, 64];
        let data: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|k| ((k % 64) as f32).sin())
            .collect();

        let x: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(shape, &data));

        let rope_result = x.rope(&cos, &sin);
        let output = rope_result.as_slice().await.unwrap();

        // Verify shapes
        assert_eq!(output.shape(), &[1, 3, 11, 64]);

        // Verify some values are within expected range (not NaN or infinity)
        for b in 0..shape[0] {
            for h in 0..shape[1] {
                for s in 0..shape[2] {
                    for d in 0..shape[3] {
                        let val = output[[b, h, s, d]];
                        assert!(
                            val.is_finite(),
                            "Non-finite value at [{}, {}, {}, {}]: {}",
                            b,
                            h,
                            s,
                            d,
                            val
                        );
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_rope_interleaved_cpu() {
        let pos_shape = [11, 32];
        let cos_data: Vec<Vec<f32>> = (0..pos_shape[0])
            .map(|i| {
                (0..pos_shape[1])
                    .map(|j| {
                        ((i as f32) / 10000f32.powf((2 * j) as f32 / (pos_shape[1] * 2) as f32))
                            .cos()
                    })
                    .collect()
            })
            .collect();
        let sin_data: Vec<Vec<f32>> = (0..pos_shape[0])
            .map(|i| {
                (0..pos_shape[1])
                    .map(|j| {
                        ((i as f32) / 10000f32.powf((2 * j) as f32 / (pos_shape[1] * 2) as f32))
                            .sin()
                    })
                    .collect()
            })
            .collect();

        let cos_flat: Vec<f32> = cos_data.iter().flatten().copied().collect();
        let sin_flat: Vec<f32> = sin_data.iter().flatten().copied().collect();
        let cos: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [pos_shape[0], pos_shape[1]],
            &cos_flat,
        ));
        let sin: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [pos_shape[0], pos_shape[1]],
            &sin_flat,
        ));

        // Input: [1, 3, 11, 64]
        let shape = [1, 3, 11, 64];
        let data: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|k| ((k % 64) as f32).sin())
            .collect();

        let x: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(shape, &data));

        let rope_result = x.rope_interleaved(&cos, &sin);
        let output = rope_result.as_slice().await.unwrap();

        // Verify shapes
        assert_eq!(output.shape(), &[1, 3, 11, 64]);

        // Verify some values are within expected range
        for b in 0..shape[0] {
            for h in 0..shape[1] {
                for s in 0..shape[2] {
                    for d in 0..shape[3] {
                        let val = output[[b, h, s, d]];
                        assert!(
                            val.is_finite(),
                            "Non-finite value at [{}, {}, {}, {}]: {}",
                            b,
                            h,
                            s,
                            d,
                            val
                        );
                    }
                }
            }
        }
    }
}
