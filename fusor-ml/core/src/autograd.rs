use rustc_hash::FxHashMap;

use crate::{
    DataType, FloatDataType, Result, Tensor,
    compute_graph::{BackwardRule, NodeIndex},
    tensor::LazyTensorData,
};

pub struct Gradients {
    gradients: FxHashMap<NodeIndex, LazyTensorData>,
}

pub struct BackwardTarget {
    node: NodeIndex,
    gradient: LazyTensorData,
}

impl Gradients {
    pub(crate) fn new(gradients: FxHashMap<NodeIndex, LazyTensorData>) -> Self {
        Self { gradients }
    }
}

impl Gradients {
    pub fn get<const R: usize, D: crate::DataType>(&self, tensor: &Tensor<R, D>) -> Option<Tensor<R, D>> {
        self.gradients
            .get(&tensor.key())
            .cloned()
            .map(Tensor::from_parts)
    }
}

impl BackwardTarget {
    pub fn wrt<const R: usize, D: DataType>(tensor: &Tensor<R, D>, gradient: Tensor<R, D>) -> Self {
        Self {
            node: tensor.key(),
            gradient: gradient.data().clone(),
        }
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn with_backwards<F>(self, backwards: F) -> Self
    where
        F: Fn(Tensor<R, D>) -> Result<Vec<BackwardTarget>> + Send + Sync + 'static,
    {
        let backward: BackwardRule = std::sync::Arc::new(move |gradient: LazyTensorData| {
            let gradient = Tensor::from_parts(gradient);
            let gradients = backwards(gradient)?;
            Ok(gradients
                .into_iter()
                .map(|target| (target.node, target.gradient))
                .collect())
        });
        self.device().compute_graph().set_backward_rule(self.key(), backward);
        self
    }
}

impl<const R: usize, D: FloatDataType> Tensor<R, D> {
    pub fn backward(&self) -> Result<Gradients> {
        if self.shape().iter().product::<usize>() != 1 {
            return Err(crate::Error::msg(
                "backward() requires a single-element tensor; use backward_with() for non-scalars",
            ));
        }

        let seed = Tensor::splat(self.device(), D::one(), *self.shape());
        self.backward_with(&seed)
    }

    pub fn backward_with(&self, seed: &Tensor<R, D>) -> Result<Gradients> {
        if self.shape() != seed.shape() {
            return Err(crate::Error::msg(format!(
                "gradient seed shape mismatch: expected {:?}, got {:?}",
                self.shape(),
                seed.shape()
            )));
        }

        let gradients = self
            .device()
            .compute_graph()
            .backward(self.key(), seed.data().clone())?;
        Ok(Gradients::new(gradients))
    }
}

#[cfg(test)]
fn assert_close(left: f32, right: f32) {
    assert!(
        (left - right).abs() < 1e-3,
        "expected {right}, got {left}"
    );
}

#[cfg(test)]
#[tokio::test]
async fn test_backward_squared_sum() {
    let device = crate::Device::test_instance();

    let x = Tensor::new(&device, &[1.0f32, 2.0, 3.0]);
    let loss: Tensor<0, f32> = (&x * &x).sum::<0>(0);

    let gradients = loss.backward().unwrap();
    let dx = gradients.get(&x).unwrap().as_slice().await.unwrap();

    assert_close(dx[[0]], 2.0);
    assert_close(dx[[1]], 4.0);
    assert_close(dx[[2]], 6.0);
}

#[cfg(test)]
#[tokio::test]
async fn test_backward_matmul_with_broadcast_bias() {
    let device = crate::Device::test_instance();

    let x = Tensor::new(&device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let w = Tensor::new(&device, &[[0.5f32], [1.0], [1.5]]);
    let b = Tensor::new(&device, &[[2.0f32]]);

    let y = x.mat_mul(&w) + &b.broadcast_as([2, 1]);
    let loss: Tensor<0, f32> = y.sum::<1>(0).sum::<0>(0);

    let gradients = loss.backward().unwrap();

    let dx = gradients.get(&x).unwrap().as_slice().await.unwrap();
    assert_close(dx[[0, 0]], 0.5);
    assert_close(dx[[0, 1]], 1.0);
    assert_close(dx[[0, 2]], 1.5);
    assert_close(dx[[1, 0]], 0.5);
    assert_close(dx[[1, 1]], 1.0);
    assert_close(dx[[1, 2]], 1.5);

    let dw = gradients.get(&w).unwrap().as_slice().await.unwrap();
    assert_close(dw[[0, 0]], 5.0);
    assert_close(dw[[1, 0]], 7.0);
    assert_close(dw[[2, 0]], 9.0);

    let db = gradients.get(&b).unwrap().as_slice().await.unwrap();
    assert_close(db[[0, 0]], 2.0);
}

#[cfg(test)]
#[tokio::test]
async fn test_backward_slice() {
    let device = crate::Device::test_instance();

    let x = Tensor::new(&device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let sliced = x.slice([0..2, 1..3]);
    let loss: Tensor<0, f32> = sliced.sum::<1>(0).sum::<0>(0);

    let gradients = loss.backward().unwrap();
    let dx = gradients.get(&x).unwrap().as_slice().await.unwrap();

    assert_close(dx[[0, 0]], 0.0);
    assert_close(dx[[0, 1]], 1.0);
    assert_close(dx[[0, 2]], 1.0);
    assert_close(dx[[1, 0]], 0.0);
    assert_close(dx[[1, 1]], 1.0);
    assert_close(dx[[1, 2]], 1.0);
}

#[cfg(test)]
#[tokio::test]
async fn test_backward_relu() {
    let device = crate::Device::test_instance();

    let x = Tensor::new(&device, &[1.0f32, -2.0, 0.0, 4.0]);
    let loss: Tensor<0, f32> = x.relu().sum::<0>(0);

    let gradients = loss.backward().unwrap();
    let dx = gradients.get(&x).unwrap().as_slice().await.unwrap();

    assert_close(dx[[0]], 1.0);
    assert_close(dx[[1]], 0.0);
    assert_close(dx[[2]], 0.0);
    assert_close(dx[[3]], 1.0);
}

#[cfg(test)]
#[tokio::test]
async fn test_with_backwards_override() {
    let device = crate::Device::test_instance();

    let x = Tensor::new(&device, &[1.0f32, 2.0]);
    let captured = x.clone();
    let y = (x.clone() + 1.0).with_backwards(move |grad| {
        Ok(vec![BackwardTarget::wrt(&captured, grad * 3.0)])
    });
    let loss: Tensor<0, f32> = y.sum::<0>(0);

    let gradients = loss.backward().unwrap();
    let dx = gradients.get(&x).unwrap().as_slice().await.unwrap();

    assert_close(dx[[0]], 3.0);
    assert_close(dx[[1]], 3.0);
}
