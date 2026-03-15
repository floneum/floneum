use std::array::from_fn;

use fusor_core::{Device, Gradients, Tensor, cache::AttentionMask};
use rand::{Rng, rngs::StdRng};

use crate::config::{
    BATCH_SIZE, BLOCK_SIZE, EPS, LEARNING_RATE, N_EMBD, N_FF, N_LAYER, VOCAB_SIZE,
};

#[derive(Clone)]
pub struct NanoChatModel {
    wte: Tensor<2, f32>,
    wpe: Tensor<2, f32>,
    blocks: [TransformerBlock; N_LAYER],
    ln_f_weight: Tensor<1, f32>,
    ln_f_bias: Tensor<1, f32>,
    lm_head: Tensor<2, f32>,
}

#[derive(Clone)]
struct TransformerBlock {
    ln_1_weight: Tensor<1, f32>,
    ln_1_bias: Tensor<1, f32>,
    attn: CausalSelfAttention,
    ln_2_weight: Tensor<1, f32>,
    ln_2_bias: Tensor<1, f32>,
    mlp: Mlp,
}

#[derive(Clone)]
struct CausalSelfAttention {
    c_attn_q: Tensor<2, f32>,
    c_attn_k: Tensor<2, f32>,
    c_attn_v: Tensor<2, f32>,
    c_proj: Tensor<2, f32>,
}

#[derive(Clone)]
struct Mlp {
    c_fc: Tensor<2, f32>,
    c_fc_bias: Tensor<1, f32>,
    c_proj: Tensor<2, f32>,
    c_proj_bias: Tensor<1, f32>,
}

impl NanoChatModel {
    pub fn new(device: &Device, rng: &mut StdRng) -> Self {
        Self {
            wte: random_matrix::<VOCAB_SIZE, N_EMBD>(device, rng, 0.08),
            wpe: random_matrix::<BLOCK_SIZE, N_EMBD>(device, rng, 0.08),
            blocks: from_fn(|_| TransformerBlock::new(device, rng)),
            ln_f_weight: ones::<N_EMBD>(device),
            ln_f_bias: zeros::<N_EMBD>(device),
            lm_head: random_matrix::<N_EMBD, VOCAB_SIZE>(device, rng, 0.08),
        }
    }

    pub fn forward<const B: usize>(
        &self,
        token_inputs: &Tensor<3, f32>,
        position_inputs: &Tensor<2, f32>,
        causal_mask: &AttentionMask<f32>,
    ) -> Tensor<3, f32> {
        let token_embeddings =
            token_inputs.mat_mul(&self.wte.broadcast_as([B, VOCAB_SIZE, N_EMBD]));
        let position_embeddings: Tensor<2, f32> = position_inputs.mat_mul(&self.wpe);
        let mut x = token_embeddings.add_(&position_embeddings.broadcast_as([B, BLOCK_SIZE, N_EMBD]));

        for block in &self.blocks {
            x = block.forward::<B>(x, causal_mask);
        }

        let x = x.layer_norm(&self.ln_f_weight, Some(&self.ln_f_bias), EPS, true);
        x.mat_mul(&self.lm_head.broadcast_as([B, N_EMBD, VOCAB_SIZE]))
    }

    pub fn step(self, gradients: &Gradients) -> Self {
        Self {
            wte: sgd_step_2d(&self.wte, gradients),
            wpe: sgd_step_2d(&self.wpe, gradients),
            blocks: self.blocks.map(|block| block.step(gradients)),
            ln_f_weight: sgd_step_1d(&self.ln_f_weight, gradients),
            ln_f_bias: sgd_step_1d(&self.ln_f_bias, gradients),
            lm_head: sgd_step_2d(&self.lm_head, gradients),
        }
    }

    pub fn num_parameters(&self) -> usize {
        tensor_len(&self.wte)
            + tensor_len(&self.wpe)
            + self
                .blocks
                .iter()
                .map(TransformerBlock::num_parameters)
                .sum::<usize>()
            + tensor_len(&self.ln_f_weight)
            + tensor_len(&self.ln_f_bias)
            + tensor_len(&self.lm_head)
    }
}

impl TransformerBlock {
    fn new(device: &Device, rng: &mut StdRng) -> Self {
        Self {
            ln_1_weight: ones::<N_EMBD>(device),
            ln_1_bias: zeros::<N_EMBD>(device),
            attn: CausalSelfAttention::new(device, rng),
            ln_2_weight: ones::<N_EMBD>(device),
            ln_2_bias: zeros::<N_EMBD>(device),
            mlp: Mlp::new(device, rng),
        }
    }

    fn forward<const B: usize>(
        &self,
        x: Tensor<3, f32>,
        causal_mask: &AttentionMask<f32>,
    ) -> Tensor<3, f32> {
        let attn_input = x.layer_norm(&self.ln_1_weight, Some(&self.ln_1_bias), EPS, true);
        let attn_output = self.attn.forward::<B>(&attn_input, causal_mask);
        let x = x + attn_output;

        let mlp_input = x.layer_norm(&self.ln_2_weight, Some(&self.ln_2_bias), EPS, true);
        x + self.mlp.forward::<B>(&mlp_input)
    }

    fn step(self, gradients: &Gradients) -> Self {
        Self {
            ln_1_weight: sgd_step_1d(&self.ln_1_weight, gradients),
            ln_1_bias: sgd_step_1d(&self.ln_1_bias, gradients),
            attn: self.attn.step(gradients),
            ln_2_weight: sgd_step_1d(&self.ln_2_weight, gradients),
            ln_2_bias: sgd_step_1d(&self.ln_2_bias, gradients),
            mlp: self.mlp.step(gradients),
        }
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.ln_1_weight)
            + tensor_len(&self.ln_1_bias)
            + self.attn.num_parameters()
            + tensor_len(&self.ln_2_weight)
            + tensor_len(&self.ln_2_bias)
            + self.mlp.num_parameters()
    }
}

impl CausalSelfAttention {
    fn new(device: &Device, rng: &mut StdRng) -> Self {
        Self {
            c_attn_q: random_matrix::<N_EMBD, N_EMBD>(device, rng, 0.08),
            c_attn_k: random_matrix::<N_EMBD, N_EMBD>(device, rng, 0.08),
            c_attn_v: random_matrix::<N_EMBD, N_EMBD>(device, rng, 0.08),
            c_proj: random_matrix::<N_EMBD, N_EMBD>(device, rng, 0.08),
        }
    }

    fn forward<const B: usize>(
        &self,
        x: &Tensor<3, f32>,
        causal_mask: &AttentionMask<f32>,
    ) -> Tensor<3, f32> {
        let q = x.mat_mul(&self.c_attn_q.broadcast_as([B, N_EMBD, N_EMBD]));
        let k = x.mat_mul(&self.c_attn_k.broadcast_as([B, N_EMBD, N_EMBD]));
        let v = x.mat_mul(&self.c_attn_v.broadcast_as([B, N_EMBD, N_EMBD]));

        let scores = q.mat_mul(&k.transpose(1, 2)) / (N_EMBD as f32).sqrt();
        let masked = causal_mask.apply(&scores);
        let weights_exp = masked.exp();
        let attention = weights_exp.div_(&weights_exp.sum_keepdim(2));

        attention
            .mat_mul(&v)
            .mat_mul(&self.c_proj.broadcast_as([B, N_EMBD, N_EMBD]))
    }

    fn step(self, gradients: &Gradients) -> Self {
        Self {
            c_attn_q: sgd_step_2d(&self.c_attn_q, gradients),
            c_attn_k: sgd_step_2d(&self.c_attn_k, gradients),
            c_attn_v: sgd_step_2d(&self.c_attn_v, gradients),
            c_proj: sgd_step_2d(&self.c_proj, gradients),
        }
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.c_attn_q)
            + tensor_len(&self.c_attn_k)
            + tensor_len(&self.c_attn_v)
            + tensor_len(&self.c_proj)
    }
}

impl Mlp {
    fn new(device: &Device, rng: &mut StdRng) -> Self {
        Self {
            c_fc: random_matrix::<N_EMBD, N_FF>(device, rng, 0.08),
            c_fc_bias: zeros::<N_FF>(device),
            c_proj: random_matrix::<N_FF, N_EMBD>(device, rng, 0.08),
            c_proj_bias: zeros::<N_EMBD>(device),
        }
    }

    fn forward<const B: usize>(&self, x: &Tensor<3, f32>) -> Tensor<3, f32> {
        let hidden = x
            .mat_mul(&self.c_fc.broadcast_as([B, N_EMBD, N_FF]))
            .add_(&self.c_fc_bias)
            .relu();

        hidden
            .mat_mul(&self.c_proj.broadcast_as([B, N_FF, N_EMBD]))
            .add_(&self.c_proj_bias)
    }

    fn step(self, gradients: &Gradients) -> Self {
        Self {
            c_fc: sgd_step_2d(&self.c_fc, gradients),
            c_fc_bias: sgd_step_1d(&self.c_fc_bias, gradients),
            c_proj: sgd_step_2d(&self.c_proj, gradients),
            c_proj_bias: sgd_step_1d(&self.c_proj_bias, gradients),
        }
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.c_fc)
            + tensor_len(&self.c_fc_bias)
            + tensor_len(&self.c_proj)
            + tensor_len(&self.c_proj_bias)
    }
}

fn random_matrix<const ROWS: usize, const COLS: usize>(
    device: &Device,
    rng: &mut StdRng,
    scale: f32,
) -> Tensor<2, f32> {
    let data: [[f32; COLS]; ROWS] = from_fn(|_| from_fn(|_| rng.random_range(-scale..scale)));
    Tensor::new(device, &data)
}

fn ones<const LEN: usize>(device: &Device) -> Tensor<1, f32> {
    Tensor::new(device, &[1.0; LEN])
}

fn zeros<const LEN: usize>(device: &Device) -> Tensor<1, f32> {
    Tensor::new(device, &[0.0; LEN])
}

fn sgd_step_1d(parameter: &Tensor<1, f32>, gradients: &Gradients) -> Tensor<1, f32> {
    let gradient = gradients.get(parameter).unwrap();
    (parameter - &(gradient * LEARNING_RATE)).detach()
}

fn sgd_step_2d(parameter: &Tensor<2, f32>, gradients: &Gradients) -> Tensor<2, f32> {
    let gradient = gradients.get(parameter).unwrap();
    (parameter - &(gradient * LEARNING_RATE)).detach()
}

fn tensor_len<const R: usize>(tensor: &Tensor<R, f32>) -> usize {
    tensor.shape().iter().product()
}

#[allow(dead_code)]
const _: usize = BATCH_SIZE;
