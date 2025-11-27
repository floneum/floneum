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
