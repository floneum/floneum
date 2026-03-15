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
async fn test_backward_matmul_with_broadcasted_weight_batch() {
    let device = crate::Device::test_instance();

    let x = Tensor::new(
        &device,
        &[
            [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
    );
    let w = Tensor::new(&device, &[[0.5f32, 1.0], [1.5, 2.0], [2.5, 3.0]]);

    let y = x.mat_mul(&w.broadcast_as([2, 3, 2]));
    let loss: Tensor<0, f32> = y.sum::<2>(2).sum::<1>(1).sum::<0>(0);

    let gradients = loss.backward().unwrap();

    let dw = gradients.get(&w).unwrap().as_slice().await.unwrap();
    assert_close(dw[[0, 0]], 22.0);
    assert_close(dw[[0, 1]], 22.0);
    assert_close(dw[[1, 0]], 26.0);
    assert_close(dw[[1, 1]], 26.0);
    assert_close(dw[[2, 0]], 30.0);
    assert_close(dw[[2, 1]], 30.0);
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

#[cfg(test)]
#[tokio::test]
async fn test_backward_after_materializing_loss_scalar() {
    let device = crate::Device::test_instance();

    let x = Tensor::new(&device, &[1.0f32, -2.0, 3.0]);
    let loss: Tensor<0, f32> = ((x.relu() + 1.0) * 2.0).sum::<0>(0);

    let loss_value = loss.to_scalar().await.unwrap();
    assert_close(loss_value, 14.0);

    let gradients = loss.backward().unwrap();
    let dx = gradients.get(&x).unwrap().as_slice().await.unwrap();

    assert_close(dx[[0]], 2.0);
    assert_close(dx[[1]], 0.0);
    assert_close(dx[[2]], 2.0);
}

#[cfg(test)]
#[tokio::test]
async fn test_backward_tiny_transformer_parameter_grads_present() {
    use crate::cache::AttentionMask;

    const VOCAB: usize = 4;
    const SEQ: usize = 3;
    const BATCH: usize = 2;
    const MODEL: usize = 4;
    const FF: usize = 6;
    const EPS: f32 = 1e-5;

    let device = crate::Device::test_instance();

    let token_inputs: Tensor<3, f32> = Tensor::new(
        &device,
        &[
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ],
    );
    let targets: Tensor<3, f32> = Tensor::new(
        &device,
        &[
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
        ],
    );
    let position_inputs: Tensor<2, f32> = Tensor::new(
        &device,
        &[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    );
    let causal_mask = AttentionMask::causal(&device, SEQ);

    let token_projection = Tensor::new(
        &device,
        &[
            [0.10, -0.02, 0.03, 0.04],
            [0.05, 0.06, -0.07, 0.08],
            [-0.04, 0.03, 0.02, -0.01],
            [0.07, -0.05, 0.06, 0.02],
        ],
    );
    let position_projection = Tensor::new(
        &device,
        &[
            [0.01, 0.02, 0.03, 0.04],
            [0.04, 0.03, 0.02, 0.01],
            [-0.02, 0.01, 0.00, 0.03],
        ],
    );
    let ln1_weight = Tensor::new(&device, &[1.0, 1.0, 1.0, 1.0]);
    let ln1_bias = Tensor::new(&device, &[0.0, 0.0, 0.0, 0.0]);
    let w_q = Tensor::new(
        &device,
        &[
            [0.02, 0.03, 0.01, -0.02],
            [0.01, -0.01, 0.04, 0.02],
            [0.05, 0.02, -0.03, 0.01],
            [-0.02, 0.01, 0.02, 0.03],
        ],
    );
    let w_k = Tensor::new(
        &device,
        &[
            [0.01, -0.03, 0.02, 0.04],
            [0.02, 0.05, -0.01, 0.03],
            [0.03, 0.01, 0.04, -0.02],
            [0.00, 0.02, 0.01, 0.05],
        ],
    );
    let w_v = Tensor::new(
        &device,
        &[
            [0.04, 0.01, -0.02, 0.03],
            [-0.01, 0.03, 0.02, 0.04],
            [0.02, 0.05, 0.01, -0.03],
            [0.03, -0.02, 0.04, 0.01],
        ],
    );
    let w_o = Tensor::new(
        &device,
        &[
            [0.03, 0.02, 0.01, -0.01],
            [0.04, -0.02, 0.03, 0.02],
            [0.01, 0.05, -0.01, 0.03],
            [0.02, 0.01, 0.04, -0.02],
        ],
    );
    let ln2_weight = Tensor::new(&device, &[1.0, 1.0, 1.0, 1.0]);
    let ln2_bias = Tensor::new(&device, &[0.0, 0.0, 0.0, 0.0]);
    let w1 = Tensor::new(
        &device,
        &[
            [0.02, 0.01, 0.03, -0.02, 0.04, 0.01],
            [0.01, 0.04, -0.01, 0.02, 0.03, 0.05],
            [0.03, -0.02, 0.05, 0.01, -0.01, 0.02],
            [0.04, 0.02, 0.01, 0.03, 0.02, -0.02],
        ],
    );
    let b1 = Tensor::new(&device, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let w2 = Tensor::new(
        &device,
        &[
            [0.01, 0.02, 0.03, 0.04],
            [0.02, 0.03, -0.01, 0.01],
            [0.04, -0.02, 0.01, 0.03],
            [0.03, 0.01, 0.02, -0.01],
            [0.01, 0.04, 0.03, 0.02],
            [0.02, -0.01, 0.04, 0.03],
        ],
    );
    let b2 = Tensor::new(&device, &[0.0, 0.0, 0.0, 0.0]);
    let ln_out_weight = Tensor::new(&device, &[1.0, 1.0, 1.0, 1.0]);
    let ln_out_bias = Tensor::new(&device, &[0.0, 0.0, 0.0, 0.0]);
    let lm_head = Tensor::new(
        &device,
        &[
            [0.02, 0.01, 0.03, 0.04],
            [0.03, 0.02, 0.01, -0.01],
            [0.04, -0.02, 0.02, 0.03],
            [0.01, 0.03, 0.04, 0.02],
        ],
    );

    let token_embeddings =
        token_inputs.mat_mul(&token_projection.broadcast_as([BATCH, VOCAB, MODEL]));
    let position_embeddings: Tensor<2, f32> = position_inputs.mat_mul(&position_projection);
    let position_embeddings_broadcast = position_embeddings.broadcast_as([BATCH, SEQ, MODEL]);
    let mut x = token_embeddings.add_(&position_embeddings_broadcast);
    let embedding_sum = x.clone();

    let attn_input = x.layer_norm(&ln1_weight, Some(&ln1_bias), EPS, true);
    let q = attn_input.mat_mul(&w_q.broadcast_as([BATCH, MODEL, MODEL]));
    let k = attn_input.mat_mul(&w_k.broadcast_as([BATCH, MODEL, MODEL]));
    let v = attn_input.mat_mul(&w_v.broadcast_as([BATCH, MODEL, MODEL]));

    let scores = q.mat_mul(&k.transpose(1, 2)) / (MODEL as f32).sqrt();
    let masked_scores = causal_mask.apply(&scores);
    let weights_exp = masked_scores.exp();
    let attention = weights_exp.div_(&weights_exp.sum_keepdim(2));
    let attention_output = attention
        .mat_mul(&v)
        .mat_mul(&w_o.broadcast_as([BATCH, MODEL, MODEL]));
    x = x + attention_output;
    let after_attention = x.clone();

    let ff_input = x.layer_norm(&ln2_weight, Some(&ln2_bias), EPS, true);
    let ff_hidden = ff_input
        .mat_mul(&w1.broadcast_as([BATCH, MODEL, FF]))
        .add_(&b1)
        .relu();
    let ff_output = ff_hidden
        .mat_mul(&w2.broadcast_as([BATCH, FF, MODEL]))
        .add_(&b2);
    x = x + ff_output;
    let after_ff = x.clone();

    let output = x.layer_norm(&ln_out_weight, Some(&ln_out_bias), EPS, true);
    let logits = output.mat_mul(&lm_head.broadcast_as([BATCH, MODEL, VOCAB]));
    let error = &logits - &targets;
    let loss: Tensor<0, f32> = (&error * &error)
        .sum::<2>(2)
        .sum::<1>(1)
        .sum::<0>(0)
        / (BATCH * SEQ * VOCAB) as f32;

    let _ = loss.to_scalar().await.unwrap();
    let gradients = loss.backward().unwrap();

    assert!(gradients.get(&token_embeddings).is_some());
    assert!(gradients.get(&embedding_sum).is_some());
    assert!(gradients.get(&attn_input).is_some());
    assert!(gradients.get(&after_attention).is_some());
    assert!(gradients.get(&ff_input).is_some());
    assert!(gradients.get(&after_ff).is_some());
    assert!(gradients.get(&output).is_some());
    assert!(gradients.get(&logits).is_some());
    assert!(gradients.get(&token_projection).is_some());
    assert!(gradients.get(&position_projection).is_some());
    assert!(gradients.get(&w_q).is_some());
    assert!(gradients.get(&w_k).is_some());
    assert!(gradients.get(&w_v).is_some());
    assert!(gradients.get(&w_o).is_some());
    assert!(gradients.get(&w1).is_some());
    assert!(gradients.get(&w2).is_some());
    assert!(gradients.get(&lm_head).is_some());
    assert!(gradients.get(&ln1_weight).is_some());
    assert!(gradients.get(&ln1_bias).is_some());
    assert!(gradients.get(&ln2_weight).is_some());
    assert!(gradients.get(&ln2_bias).is_some());
    assert!(gradients.get(&b1).is_some());
    assert!(gradients.get(&b2).is_some());
    assert!(gradients.get(&ln_out_weight).is_some());
    assert!(gradients.get(&ln_out_bias).is_some());
}
