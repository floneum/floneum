use crate::{D, DataType, Tensor};

fn rotate_half<const N: usize, T: DataType>(xs: Tensor<N, T>) -> Tensor<N, T> {
    let last_dim = xs.shape().last().unwrap();
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2);
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2);
    Tensor::cat([-xs2, xs1], D::Minus1)
}

impl<T: DataType> Tensor<4, T> {
    pub fn rope(&self, cos: &Tensor<2, T>, sin: &Tensor<2, T>) -> Tensor<4, T> {
        let [_, _, sequence_length, _] = *self.shape();

        let cos = Tensor::cat([cos.clone(), cos.clone()], D::Minus1);
        let sin = Tensor::cat([sin.clone(), sin.clone()], D::Minus1);

        let cos = cos.narrow(0, 0, sequence_length);
        let sin = sin.narrow(0, 0, sequence_length);

        let cos = cos.unsqueeze(0).unsqueeze(0);
        let sin = sin.unsqueeze(0).unsqueeze(0);

        let rotated = rotate_half(self.clone());
        self.mul_(&cos) + rotated.mul_(&sin)
    }

    // let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    // let cos = cos
    //     .narrow(0, 0, seq_len)?
    //     .reshape((seq_len, n_embd / 2, 1))?;
    // let sin = sin
    //     .narrow(0, 0, seq_len)?
    //     .reshape((seq_len, n_embd / 2, 1))?;
    // let cos = cos.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
    // let sin = sin.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
    // let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    // let x0 = x.narrow(D::Minus1, 0, 1)?;
    // let x1 = x.narrow(D::Minus1, 1, 1)?;
    // let y0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    // let y1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    // let rope = Tensor::cat(&[y0, y1], D::Minus1)?;
    // let rope = rope.flatten_from(D::Minus2)?;
    // Ok(rope)
    pub fn rope_interleaved(&self, cos: &Tensor<2, T>, sin: &Tensor<2, T>) -> Tensor<4, T> {
        let [bz, n_head, sequence_length, embed] = *self.shape();

        let cos = cos
            .narrow(0, 0, sequence_length)
            .reshape([sequence_length, embed / 2, 1])
            .broadcast_as([bz, 1, sequence_length, embed / 2, 1]);
        let sin = sin
            .narrow(0, 0, sequence_length)
            .reshape([sequence_length, embed / 2, 1])
            .broadcast_as([bz, 1, sequence_length, embed / 2, 1]);
        let x = self.reshape([bz, n_head, sequence_length, embed / 2, 2]);

        let x0 = x.narrow(D::Minus1, 0, 1);
        let x1 = x.narrow(D::Minus1, 1, 1);

        let y0 = &x0.mul_(&cos) - &x1.mul_(&sin);
        let y1 = &x0.mul_(&sin) + &x1.mul_(&cos);

        Tensor::cat([y0, y1], D::Minus1).flatten_last_n::<1, _>()
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_rope_interleaved() {
    use candle_core::IndexOp;

    let candle_device = candle_core::Device::Cpu;
    let device = crate::Device::new().await.unwrap();

    let pos_shape = [11, 32];
    let cos = (0..pos_shape[0])
        .map(|i| {
            (0..pos_shape[1])
                .map(|j| {
                    ((i as f32) / 10000f32.powf((2 * (j / 2)) as f32 / pos_shape[1] as f32)).cos()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let sin = (0..pos_shape[0])
        .map(|i| {
            (0..pos_shape[1])
                .map(|j| {
                    ((i as f32) / 10000f32.powf((2 * (j / 2)) as f32 / pos_shape[1] as f32)).sin()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let fusor_cos = Tensor::new(&device, &cos);
    let fusor_sin = Tensor::new(&device, &sin);
    let candle_cos = candle_core::Tensor::new(cos, &candle_device).unwrap();
    let candle_sin = candle_core::Tensor::new(sin, &candle_device).unwrap();

    let shape = [1, 3, 11, 64];
    let data: Vec<Vec<Vec<Vec<f32>>>> = (0..shape[0])
        .map(|_| {
            (0..shape[1])
                .map(|_| {
                    (0..shape[2])
                        .map(|_| {
                            use rand::Rng;

                            (0..shape[3])
                                .map(|_| rand::rng().random_range(-1.0..1.0))
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let fusor_tensor = Tensor::new(&device, &data);
    let candle_tensor = candle_core::Tensor::new(data, &candle_device).unwrap();
    let fusor_rope = fusor_tensor.rope_interleaved(&fusor_cos, &fusor_sin);
    let candle_rope =
        candle_nn::rotary_emb::rope_i(&candle_tensor, &candle_cos, &candle_sin).unwrap();

    let fusor_as_slice = fusor_rope.i((0, .., .., ..)).as_slice().await.unwrap();
    let candle_as_slice = candle_rope
        .i((0, .., .., ..))
        .unwrap()
        .to_vec3::<f32>()
        .unwrap();

    for i in 0..shape[1] {
        for j in 0..shape[2] {
            for k in 0..shape[3] {
                let a = fusor_as_slice[[i, j, k]];
                let b = candle_as_slice[i][j][k];
                assert!((a - b).abs() < 1e-4, "mismatch at {i},{j},{k}: {a} vs {b}");
            }
        }
    }
}
