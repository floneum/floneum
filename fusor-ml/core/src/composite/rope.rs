use crate::{DataType, Tensor};

const LAST_OF_THREE_DIMS: usize = 2;

fn rotate_half<const N: usize, D: DataType>(xs: Tensor<N, D>) -> Tensor<N, D> {
    let last_dim = xs.shape().last().unwrap();
    let xs1 = xs.narrow(N - 1, 0, last_dim / 2);
    let xs2 = xs.narrow(N - 1, last_dim / 2, last_dim - last_dim / 2);
    Tensor::cat([-xs2, xs1], N - 1)
}

impl<D: DataType> Tensor<3, D> {
    pub fn rope(self, cos: Tensor<2, D>, sin: Tensor<2, D>) -> Tensor<3, D> {
        let shape = *self.shape();
        let [_height, sequence_length, _embed] = shape;
        let cos = Tensor::cat([cos.clone(), cos.clone()], LAST_OF_THREE_DIMS);
        let sin = Tensor::cat([sin.clone(), sin.clone()], LAST_OF_THREE_DIMS);
        let cos = cos.narrow(0, 0, sequence_length);
        let sin = sin.narrow(0, 0, sequence_length);
        let rotated = rotate_half(self.clone());
        self * cos.broadcast(shape) + rotated * sin.broadcast(shape)
    }

    pub fn rope_interleaved(self, cos: Tensor<2, D>, sin: Tensor<2, D>) -> Tensor<3, D> {
        let shape = *self.shape();
        let [height, sequence_length, embed] = shape;
        println!("height: {height}, sequence_length: {sequence_length}, embed: {embed}");

        println!("cos: {:?}", cos);
        println!("sin: {:?}", sin);

        let cos = cos
            .narrow(0, 0, sequence_length)
            .reshape([1, sequence_length, embed / 2, 1]);
        let sin = sin
            .narrow(0, 0, sequence_length)
            .reshape([1, sequence_length, embed / 2, 1]);

        let x = self.reshape([height, sequence_length, embed / 2, 2]);

        let x0 = x.narrow(LAST_OF_THREE_DIMS, 0, 1);
        let x1 = x.narrow(LAST_OF_THREE_DIMS, 1, 1);

        let y0 = &x0 * &cos - &x1 * &sin;
        let y1 = &x0 * &sin + &x1 * &cos;

        let rope = Tensor::cat([y0, y1], LAST_OF_THREE_DIMS);
        let rope = rope.reshape([height, sequence_length, embed]);
        rope
    }
}
