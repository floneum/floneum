use crate::{DataType, Tensor};

fn rotate_half<const N: usize, D: DataType>(xs: Tensor<N, D>) -> Tensor<N, D> {
    let last_dim = xs.shape().last().unwrap();
    let xs1 = xs.narrow(N - 1, 0, last_dim / 2);
    let xs2 = xs.narrow(N - 1, last_dim / 2, last_dim - last_dim / 2);
    Tensor::cat([-xs2, xs1], N - 1)
}

impl<D: DataType> Tensor<3, D> {
    pub fn rope(self, cos: Tensor<2, D>, sin: Tensor<2, D>) -> Tensor<3, D> {
        const LAST_DIM: usize = 2;
        let shape = *self.shape();
        let [_height, sequence_length, _embed] = shape;
        let cos = Tensor::cat([cos.clone(), cos.clone()], LAST_DIM);
        let sin = Tensor::cat([sin.clone(), sin.clone()], LAST_DIM);
        let cos = cos.narrow(0, 0, sequence_length);
        let sin = sin.narrow(0, 0, sequence_length);
        let rotated = rotate_half(self.clone());
        self * cos.broadcast_as(shape) + rotated * sin.broadcast_as(shape)
    }

    pub fn rope_interleaved(self, cos: Tensor<2, D>, sin: Tensor<2, D>) -> Tensor<3, D> {
        const LAST_DIM: usize = 3;
        let shape = *self.shape();
        let [height, sequence_length, embed] = shape;

        let cos = cos
            .narrow(0, 0, sequence_length)
            .reshape([sequence_length, embed / 2, 1])
            .broadcast_as([height, sequence_length, embed / 2, 1]);
        let sin = sin
            .narrow(0, 0, sequence_length)
            .reshape([sequence_length, embed / 2, 1])
            .broadcast_as([height, sequence_length, embed / 2, 1]);

        let x = self.reshape([height, sequence_length, embed / 2, 2]);

        let x0 = x.narrow(LAST_DIM, 0, 1);
        let x1 = x.narrow(LAST_DIM, 1, 1);

        let a = &x0 * &cos;
        let b = &x1 * &sin;
        let y0 = a - b;
        let y1 = &x0 * &sin + &x1 * &cos;

        let rope = Tensor::cat([y0, y1], LAST_DIM);

        rope.reshape([height, sequence_length, embed])
    }
}
