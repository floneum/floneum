use fusor::{
    Device,
    Tensor as RawTensor,
    autograd::{Gradients, Graph, Tensor},
    ToVec1, ToVec2,
};
use rand::{Rng, rngs::StdRng};

use crate::config::RuntimeConfig;

pub struct NamedTensor {
    pub name: String,
    pub shape: Vec<u32>,
    pub data: Vec<u8>,
}

#[derive(Clone, Copy)]
struct ModelShape {
    block_size: usize,
    n_embd: usize,
    n_head: usize,
    n_ff: usize,
    eps: f32,
}

#[derive(Clone)]
pub struct CpuNanoChatModel {
    graph: Graph,
    shape: ModelShape,
    attention_period: usize,
    vocab_size: usize,
    wte: Tensor<2>,
    wpe: Tensor<2>,
    blocks: Vec<TransformerBlock>,
    ln_f_weight: Tensor<1>,
    ln_f_bias: Tensor<1>,
    lm_head: Tensor<2>,
}

pub struct CpuAdamW {
    step: usize,
    settings: AdamWSettings,
    wte: AdamMoments<2>,
    wpe: AdamMoments<2>,
    blocks: Vec<AdamBlockState>,
    ln_f_weight: AdamMoments<1>,
    ln_f_bias: AdamMoments<1>,
    lm_head: AdamMoments<2>,
}

#[derive(Clone)]
struct TransformerBlock {
    ln_1_weight: Tensor<1>,
    ln_1_bias: Tensor<1>,
    mixer: SequenceMixer,
    ln_2_weight: Tensor<1>,
    ln_2_bias: Tensor<1>,
    mlp: Mlp,
}

#[derive(Clone)]
enum SequenceMixer {
    Attention(CausalSelfAttention),
    Recurrent(RecurrentMixer),
}

#[derive(Clone)]
struct CausalSelfAttention {
    c_attn_q: Tensor<2>,
    c_attn_k: Tensor<2>,
    c_attn_v: Tensor<2>,
    c_proj: Tensor<2>,
}

#[derive(Clone)]
struct RecurrentMixer {
    input_proj: Tensor<2>,
    state_proj: Tensor<2>,
    gate_input_proj: Tensor<2>,
    gate_state_proj: Tensor<2>,
    out_proj: Tensor<2>,
}

#[derive(Clone)]
struct Mlp {
    c_fc: Tensor<2>,
    c_fc_bias: Tensor<1>,
    c_proj: Tensor<2>,
    c_proj_bias: Tensor<1>,
}

struct AdamBlockState {
    ln_1_weight: AdamMoments<1>,
    ln_1_bias: AdamMoments<1>,
    mixer: AdamMixerState,
    ln_2_weight: AdamMoments<1>,
    ln_2_bias: AdamMoments<1>,
    mlp: AdamMlpState,
}

enum AdamMixerState {
    Attention(AdamAttentionState),
    Recurrent(AdamRecurrentState),
}

struct AdamAttentionState {
    c_attn_q: AdamMoments<2>,
    c_attn_k: AdamMoments<2>,
    c_attn_v: AdamMoments<2>,
    c_proj: AdamMoments<2>,
}

struct AdamRecurrentState {
    input_proj: AdamMoments<2>,
    state_proj: AdamMoments<2>,
    gate_input_proj: AdamMoments<2>,
    gate_state_proj: AdamMoments<2>,
    out_proj: AdamMoments<2>,
}

struct AdamMlpState {
    c_fc: AdamMoments<2>,
    c_fc_bias: AdamMoments<1>,
    c_proj: AdamMoments<2>,
    c_proj_bias: AdamMoments<1>,
}

struct AdamMoments<const R: usize> {
    m: RawTensor<R, f32>,
    v: RawTensor<R, f32>,
}

#[derive(Clone, Copy)]
struct AdamWSettings {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    adam_eps: f32,
    weight_decay: f32,
}

impl CpuNanoChatModel {
    pub fn new(device: &Device, rng: &mut StdRng, vocab_size: usize, config: &RuntimeConfig) -> Self {
        let graph = Graph::new();
        let shape = ModelShape {
            block_size: config.block_size,
            n_embd: config.n_embd,
            n_head: config.n_head.max(1),
            n_ff: config.n_ff,
            eps: config.eps,
        };
        assert_eq!(
            shape.n_embd % shape.n_head,
            0,
            "NANOCHAT_N_EMBD ({}) must be divisible by NANOCHAT_N_HEAD ({})",
            shape.n_embd,
            shape.n_head
        );

        Self {
            graph: graph.clone(),
            shape,
            attention_period: config.attention_period.max(1),
            vocab_size,
            wte: random_matrix(&graph, device, rng, vocab_size, shape.n_embd, config.init_scale),
            wpe: random_matrix(&graph, device, rng, shape.block_size, shape.n_embd, config.init_scale),
            blocks: (0..config.n_layer)
                .map(|index| {
                    TransformerBlock::new(
                        &graph,
                        device,
                        rng,
                        shape,
                        config.init_scale,
                        is_attention_layer(index, config.attention_period.max(1)),
                    )
                })
                .collect(),
            ln_f_weight: ones(&graph, device, shape.n_embd),
            ln_f_bias: zeros(&graph, device, shape.n_embd),
            lm_head: random_matrix(&graph, device, rng, shape.n_embd, vocab_size, config.init_scale),
        }
    }

    pub fn num_parameters(&self) -> usize {
        tensor_len(&self.wte)
            + tensor_len(&self.wpe)
            + self.blocks.iter().map(TransformerBlock::num_parameters).sum::<usize>()
            + tensor_len(&self.ln_f_weight)
            + tensor_len(&self.ln_f_bias)
            + tensor_len(&self.lm_head)
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn block_size(&self) -> usize {
        self.shape.block_size
    }

    pub fn n_embd(&self) -> usize {
        self.shape.n_embd
    }

    pub fn n_head(&self) -> usize {
        self.shape.n_head
    }

    pub fn n_ff(&self) -> usize {
        self.shape.n_ff
    }

    pub fn n_layer(&self) -> usize {
        self.blocks.len()
    }

    pub fn eps(&self) -> f32 {
        self.shape.eps
    }

    pub fn attention_period(&self) -> usize {
        self.attention_period
    }

    pub fn forward(
        &self,
        token_inputs: &RawTensor<2, u32>,
        position_inputs: &RawTensor<2, u32>,
        causal_mask: &Tensor<3>,
    ) -> Tensor<3> {
        let batch_size = token_inputs.shape()[0];
        let token_embeddings = self.wte.embedding(token_inputs);
        let position_embeddings = self.wpe.embedding(position_inputs);
        let mut x = token_embeddings.add(&position_embeddings);

        for block in &self.blocks {
            x = block.forward(x, causal_mask, batch_size, self.shape);
        }

        let x = x.layer_norm(&self.ln_f_weight, Some(&self.ln_f_bias), self.shape.eps);
        x.mat_mul(&self.lm_head.broadcast_as([batch_size, self.shape.n_embd, self.vocab_size]))
    }

    pub async fn named_tensors(&self) -> Vec<NamedTensor> {
        let mut tensors = Vec::new();
        push_tensor_2d(&mut tensors, "token_embd.weight", &self.wte).await;
        push_tensor_2d(&mut tensors, "position_embd.weight", &self.wpe).await;

        for (index, block) in self.blocks.iter().enumerate() {
            block.append_named_tensors(index, &mut tensors).await;
        }

        push_tensor_1d(&mut tensors, "output_norm.weight", &self.ln_f_weight).await;
        push_tensor_1d(&mut tensors, "output_norm.bias", &self.ln_f_bias).await;
        push_tensor_2d(&mut tensors, "output.weight", &self.lm_head).await;
        tensors
    }
}

impl CpuAdamW {
    pub fn new(device: &Device, model: &CpuNanoChatModel, config: &RuntimeConfig) -> Self {
        Self {
            step: 0,
            settings: AdamWSettings {
                learning_rate: config.learning_rate,
                beta1: config.beta1,
                beta2: config.beta2,
                adam_eps: config.adam_eps,
                weight_decay: config.weight_decay,
            },
            wte: AdamMoments::zeros_like(device, &model.wte),
            wpe: AdamMoments::zeros_like(device, &model.wpe),
            blocks: model.blocks.iter().map(|block| AdamBlockState::new(device, block)).collect(),
            ln_f_weight: AdamMoments::zeros_like(device, &model.ln_f_weight),
            ln_f_bias: AdamMoments::zeros_like(device, &model.ln_f_bias),
            lm_head: AdamMoments::zeros_like(device, &model.lm_head),
        }
    }

    pub fn step(&mut self, model: CpuNanoChatModel, gradients: &Gradients) -> CpuNanoChatModel {
        self.step += 1;
        let step = self.step;
        let settings = self.settings;

        let CpuNanoChatModel {
            graph,
            shape,
            attention_period,
            vocab_size,
            wte,
            wpe,
            blocks,
            ln_f_weight,
            ln_f_bias,
            lm_head,
        } = model;

        let blocks = blocks
            .into_iter()
            .zip(self.blocks.iter_mut())
            .map(|(block, state)| state.step(block, gradients, step, settings))
            .collect();

        CpuNanoChatModel {
            graph,
            shape,
            attention_period,
            vocab_size,
            wte: adamw_update(&wte, &mut self.wte, gradients, step, settings),
            wpe: adamw_update(&wpe, &mut self.wpe, gradients, step, settings),
            blocks,
            ln_f_weight: adamw_update(&ln_f_weight, &mut self.ln_f_weight, gradients, step, settings),
            ln_f_bias: adamw_update(&ln_f_bias, &mut self.ln_f_bias, gradients, step, settings),
            lm_head: adamw_update(&lm_head, &mut self.lm_head, gradients, step, settings),
        }
    }

    pub fn set_learning_rate(&mut self, learning_rate: f32) {
        self.settings.learning_rate = learning_rate;
    }
}

impl TransformerBlock {
    fn new(
        graph: &Graph,
        device: &Device,
        rng: &mut StdRng,
        shape: ModelShape,
        init_scale: f32,
        is_attention_layer: bool,
    ) -> Self {
        Self {
            ln_1_weight: ones(graph, device, shape.n_embd),
            ln_1_bias: zeros(graph, device, shape.n_embd),
            mixer: if is_attention_layer {
                SequenceMixer::Attention(CausalSelfAttention::new(graph, device, rng, shape, init_scale))
            } else {
                SequenceMixer::Recurrent(RecurrentMixer::new(graph, device, rng, shape, init_scale))
            },
            ln_2_weight: ones(graph, device, shape.n_embd),
            ln_2_bias: zeros(graph, device, shape.n_embd),
            mlp: Mlp::new(graph, device, rng, shape, init_scale),
        }
    }

    fn forward(&self, x: Tensor<3>, causal_mask: &Tensor<3>, batch_size: usize, shape: ModelShape) -> Tensor<3> {
        let attn_input = x.layer_norm(&self.ln_1_weight, Some(&self.ln_1_bias), shape.eps);
        let attn_output = self.mixer.forward(&attn_input, causal_mask, batch_size, shape);
        let x = x.add(&attn_output);
        let mlp_input = x.layer_norm(&self.ln_2_weight, Some(&self.ln_2_bias), shape.eps);
        x.add(&self.mlp.forward(&mlp_input, batch_size, shape))
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.ln_1_weight)
            + tensor_len(&self.ln_1_bias)
            + self.mixer.num_parameters()
            + tensor_len(&self.ln_2_weight)
            + tensor_len(&self.ln_2_bias)
            + self.mlp.num_parameters()
    }

    async fn append_named_tensors(&self, index: usize, tensors: &mut Vec<NamedTensor>) {
        let prefix = format!("blk.{index}");
        push_tensor_1d(tensors, &format!("{prefix}.ln_1.weight"), &self.ln_1_weight).await;
        push_tensor_1d(tensors, &format!("{prefix}.ln_1.bias"), &self.ln_1_bias).await;
        self.mixer.append_named_tensors(&prefix, tensors).await;
        push_tensor_1d(tensors, &format!("{prefix}.ln_2.weight"), &self.ln_2_weight).await;
        push_tensor_1d(tensors, &format!("{prefix}.ln_2.bias"), &self.ln_2_bias).await;
        self.mlp.append_named_tensors(&prefix, tensors).await;
    }
}

impl CausalSelfAttention {
    fn new(graph: &Graph, device: &Device, rng: &mut StdRng, shape: ModelShape, init_scale: f32) -> Self {
        Self {
            c_attn_q: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
            c_attn_k: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
            c_attn_v: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
            c_proj: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
        }
    }

    fn forward(&self, x: &Tensor<3>, causal_mask: &Tensor<3>, batch_size: usize, shape: ModelShape) -> Tensor<3> {
        let head_dim = shape.head_dim();
        let q = x.mat_mul(&self.c_attn_q.broadcast_as([batch_size, shape.n_embd, shape.n_embd]));
        let k = x.mat_mul(&self.c_attn_k.broadcast_as([batch_size, shape.n_embd, shape.n_embd]));
        let v = x.mat_mul(&self.c_attn_v.broadcast_as([batch_size, shape.n_embd, shape.n_embd]));
        let heads = (0..shape.n_head)
            .map(|head| {
                let start = head * head_dim;
                let end = start + head_dim;
                let q_head = q.slice([0..batch_size, 0..q.shape()[1], start..end]);
                let k_head = k.slice([0..batch_size, 0..k.shape()[1], start..end]);
                let v_head = v.slice([0..batch_size, 0..v.shape()[1], start..end]);
                let scores = q_head
                    .mat_mul(&k_head.transpose(1, 2))
                    .div_scalar((head_dim as f32).sqrt());
                let masked = scores.add(causal_mask);
                let weights_exp = masked.exp();
                let attention =
                    weights_exp.div(&weights_exp.sum_keepdim(2).broadcast_as(weights_exp.shape()));
                attention.mat_mul(&v_head)
            })
            .collect::<Vec<_>>();
        Tensor::cat(heads, 2)
            .mat_mul(&self.c_proj.broadcast_as([batch_size, shape.n_embd, shape.n_embd]))
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.c_attn_q)
            + tensor_len(&self.c_attn_k)
            + tensor_len(&self.c_attn_v)
            + tensor_len(&self.c_proj)
    }

    async fn append_named_tensors(&self, prefix: &str, tensors: &mut Vec<NamedTensor>) {
        push_tensor_2d(tensors, &format!("{prefix}.attn_q.weight"), &self.c_attn_q).await;
        push_tensor_2d(tensors, &format!("{prefix}.attn_k.weight"), &self.c_attn_k).await;
        push_tensor_2d(tensors, &format!("{prefix}.attn_v.weight"), &self.c_attn_v).await;
        push_tensor_2d(tensors, &format!("{prefix}.attn_proj.weight"), &self.c_proj).await;
    }
}

impl RecurrentMixer {
    fn new(graph: &Graph, device: &Device, rng: &mut StdRng, shape: ModelShape, init_scale: f32) -> Self {
        Self {
            input_proj: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
            state_proj: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
            gate_input_proj: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
            gate_state_proj: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
            out_proj: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
        }
    }

    fn forward(&self, x: &Tensor<3>, batch_size: usize, shape: ModelShape) -> Tensor<3> {
        let device = x.device();
        let seq_len = x.shape()[1];
        let graph = x.graph();
        let ones = Tensor::splat(&graph, &device, 1.0, [batch_size, shape.n_embd]);
        let mut state = Tensor::zeros(&graph, &device, [batch_size, shape.n_embd]);
        let mut outputs = Vec::with_capacity(seq_len);

        for position in 0..seq_len {
            let prev_state = state.clone();
            let x_t = x
                .slice([0..batch_size, position..position + 1, 0..shape.n_embd])
                .squeeze(1);
            let candidate = x_t
                .mat_mul(&self.input_proj)
                .add(&prev_state.mat_mul(&self.state_proj))
                .tanh();
            let gate_pre = x_t
                .mat_mul(&self.gate_input_proj)
                .add(&prev_state.mat_mul(&self.gate_state_proj));
            let gate = gate_pre.tanh().add_scalar(1.0).mul_scalar(0.5);
            let keep = ones.sub(&gate);
            state = gate.mul(&candidate).add(&keep.mul(&prev_state));
            outputs.push(state.mat_mul(&self.out_proj).unsqueeze(1));
        }

        Tensor::cat(outputs, 1)
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.input_proj)
            + tensor_len(&self.state_proj)
            + tensor_len(&self.gate_input_proj)
            + tensor_len(&self.gate_state_proj)
            + tensor_len(&self.out_proj)
    }

    async fn append_named_tensors(&self, prefix: &str, tensors: &mut Vec<NamedTensor>) {
        push_tensor_2d(tensors, &format!("{prefix}.recurrent_in.weight"), &self.input_proj).await;
        push_tensor_2d(tensors, &format!("{prefix}.recurrent_state.weight"), &self.state_proj).await;
        push_tensor_2d(tensors, &format!("{prefix}.recurrent_gate_in.weight"), &self.gate_input_proj)
            .await;
        push_tensor_2d(
            tensors,
            &format!("{prefix}.recurrent_gate_state.weight"),
            &self.gate_state_proj,
        )
        .await;
        push_tensor_2d(tensors, &format!("{prefix}.recurrent_out.weight"), &self.out_proj).await;
    }
}

impl SequenceMixer {
    fn forward(&self, x: &Tensor<3>, causal_mask: &Tensor<3>, batch_size: usize, shape: ModelShape) -> Tensor<3> {
        match self {
            SequenceMixer::Attention(attn) => attn.forward(x, causal_mask, batch_size, shape),
            SequenceMixer::Recurrent(recurrent) => recurrent.forward(x, batch_size, shape),
        }
    }

    fn num_parameters(&self) -> usize {
        match self {
            SequenceMixer::Attention(attn) => attn.num_parameters(),
            SequenceMixer::Recurrent(recurrent) => recurrent.num_parameters(),
        }
    }

    async fn append_named_tensors(&self, prefix: &str, tensors: &mut Vec<NamedTensor>) {
        match self {
            SequenceMixer::Attention(attn) => attn.append_named_tensors(prefix, tensors).await,
            SequenceMixer::Recurrent(recurrent) => recurrent.append_named_tensors(prefix, tensors).await,
        }
    }
}

impl Mlp {
    fn new(graph: &Graph, device: &Device, rng: &mut StdRng, shape: ModelShape, init_scale: f32) -> Self {
        Self {
            c_fc: random_matrix(graph, device, rng, shape.n_embd, shape.n_ff, init_scale),
            c_fc_bias: zeros(graph, device, shape.n_ff),
            c_proj: random_matrix(graph, device, rng, shape.n_ff, shape.n_embd, init_scale),
            c_proj_bias: zeros(graph, device, shape.n_embd),
        }
    }

    fn forward(&self, x: &Tensor<3>, batch_size: usize, shape: ModelShape) -> Tensor<3> {
        let hidden = x
            .mat_mul(&self.c_fc.broadcast_as([batch_size, shape.n_embd, shape.n_ff]))
            .add(&self.c_fc_bias.broadcast_as([batch_size, shape.block_size, shape.n_ff]))
            .relu();

        hidden
            .mat_mul(&self.c_proj.broadcast_as([batch_size, shape.n_ff, shape.n_embd]))
            .add(&self.c_proj_bias.broadcast_as([batch_size, shape.block_size, shape.n_embd]))
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.c_fc)
            + tensor_len(&self.c_fc_bias)
            + tensor_len(&self.c_proj)
            + tensor_len(&self.c_proj_bias)
    }

    async fn append_named_tensors(&self, prefix: &str, tensors: &mut Vec<NamedTensor>) {
        push_tensor_2d(tensors, &format!("{prefix}.mlp_fc.weight"), &self.c_fc).await;
        push_tensor_1d(tensors, &format!("{prefix}.mlp_fc.bias"), &self.c_fc_bias).await;
        push_tensor_2d(tensors, &format!("{prefix}.mlp_proj.weight"), &self.c_proj).await;
        push_tensor_1d(tensors, &format!("{prefix}.mlp_proj.bias"), &self.c_proj_bias).await;
    }
}

impl AdamBlockState {
    fn new(device: &Device, block: &TransformerBlock) -> Self {
        Self {
            ln_1_weight: AdamMoments::zeros_like(device, &block.ln_1_weight),
            ln_1_bias: AdamMoments::zeros_like(device, &block.ln_1_bias),
            mixer: AdamMixerState::new(device, &block.mixer),
            ln_2_weight: AdamMoments::zeros_like(device, &block.ln_2_weight),
            ln_2_bias: AdamMoments::zeros_like(device, &block.ln_2_bias),
            mlp: AdamMlpState::new(device, &block.mlp),
        }
    }

    fn step(&mut self, block: TransformerBlock, gradients: &Gradients, step: usize, settings: AdamWSettings) -> TransformerBlock {
        TransformerBlock {
            ln_1_weight: adamw_update(&block.ln_1_weight, &mut self.ln_1_weight, gradients, step, settings),
            ln_1_bias: adamw_update(&block.ln_1_bias, &mut self.ln_1_bias, gradients, step, settings),
            mixer: self.mixer.step(block.mixer, gradients, step, settings),
            ln_2_weight: adamw_update(&block.ln_2_weight, &mut self.ln_2_weight, gradients, step, settings),
            ln_2_bias: adamw_update(&block.ln_2_bias, &mut self.ln_2_bias, gradients, step, settings),
            mlp: self.mlp.step(block.mlp, gradients, step, settings),
        }
    }
}

impl AdamMixerState {
    fn new(device: &Device, mixer: &SequenceMixer) -> Self {
        match mixer {
            SequenceMixer::Attention(attn) => Self::Attention(AdamAttentionState::new(device, attn)),
            SequenceMixer::Recurrent(recurrent) => Self::Recurrent(AdamRecurrentState::new(device, recurrent)),
        }
    }

    fn step(&mut self, mixer: SequenceMixer, gradients: &Gradients, step: usize, settings: AdamWSettings) -> SequenceMixer {
        match (self, mixer) {
            (AdamMixerState::Attention(state), SequenceMixer::Attention(attn)) => {
                SequenceMixer::Attention(state.step(attn, gradients, step, settings))
            }
            (AdamMixerState::Recurrent(state), SequenceMixer::Recurrent(recurrent)) => {
                SequenceMixer::Recurrent(state.step(recurrent, gradients, step, settings))
            }
            _ => unreachable!("mixer schedule changed after optimizer init"),
        }
    }
}

impl AdamAttentionState {
    fn new(device: &Device, attn: &CausalSelfAttention) -> Self {
        Self {
            c_attn_q: AdamMoments::zeros_like(device, &attn.c_attn_q),
            c_attn_k: AdamMoments::zeros_like(device, &attn.c_attn_k),
            c_attn_v: AdamMoments::zeros_like(device, &attn.c_attn_v),
            c_proj: AdamMoments::zeros_like(device, &attn.c_proj),
        }
    }

    fn step(&mut self, attn: CausalSelfAttention, gradients: &Gradients, step: usize, settings: AdamWSettings) -> CausalSelfAttention {
        CausalSelfAttention {
            c_attn_q: adamw_update(&attn.c_attn_q, &mut self.c_attn_q, gradients, step, settings),
            c_attn_k: adamw_update(&attn.c_attn_k, &mut self.c_attn_k, gradients, step, settings),
            c_attn_v: adamw_update(&attn.c_attn_v, &mut self.c_attn_v, gradients, step, settings),
            c_proj: adamw_update(&attn.c_proj, &mut self.c_proj, gradients, step, settings),
        }
    }
}

impl AdamRecurrentState {
    fn new(device: &Device, recurrent: &RecurrentMixer) -> Self {
        Self {
            input_proj: AdamMoments::zeros_like(device, &recurrent.input_proj),
            state_proj: AdamMoments::zeros_like(device, &recurrent.state_proj),
            gate_input_proj: AdamMoments::zeros_like(device, &recurrent.gate_input_proj),
            gate_state_proj: AdamMoments::zeros_like(device, &recurrent.gate_state_proj),
            out_proj: AdamMoments::zeros_like(device, &recurrent.out_proj),
        }
    }

    fn step(&mut self, recurrent: RecurrentMixer, gradients: &Gradients, step: usize, settings: AdamWSettings) -> RecurrentMixer {
        RecurrentMixer {
            input_proj: adamw_update(&recurrent.input_proj, &mut self.input_proj, gradients, step, settings),
            state_proj: adamw_update(&recurrent.state_proj, &mut self.state_proj, gradients, step, settings),
            gate_input_proj: adamw_update(&recurrent.gate_input_proj, &mut self.gate_input_proj, gradients, step, settings),
            gate_state_proj: adamw_update(&recurrent.gate_state_proj, &mut self.gate_state_proj, gradients, step, settings),
            out_proj: adamw_update(&recurrent.out_proj, &mut self.out_proj, gradients, step, settings),
        }
    }
}

impl AdamMlpState {
    fn new(device: &Device, mlp: &Mlp) -> Self {
        Self {
            c_fc: AdamMoments::zeros_like(device, &mlp.c_fc),
            c_fc_bias: AdamMoments::zeros_like(device, &mlp.c_fc_bias),
            c_proj: AdamMoments::zeros_like(device, &mlp.c_proj),
            c_proj_bias: AdamMoments::zeros_like(device, &mlp.c_proj_bias),
        }
    }

    fn step(&mut self, mlp: Mlp, gradients: &Gradients, step: usize, settings: AdamWSettings) -> Mlp {
        Mlp {
            c_fc: adamw_update(&mlp.c_fc, &mut self.c_fc, gradients, step, settings),
            c_fc_bias: adamw_update(&mlp.c_fc_bias, &mut self.c_fc_bias, gradients, step, settings),
            c_proj: adamw_update(&mlp.c_proj, &mut self.c_proj, gradients, step, settings),
            c_proj_bias: adamw_update(&mlp.c_proj_bias, &mut self.c_proj_bias, gradients, step, settings),
        }
    }
}

impl<const R: usize> AdamMoments<R> {
    fn zeros_like(device: &Device, parameter: &Tensor<R>) -> Self {
        let shape = parameter.shape();
        Self {
            m: RawTensor::zeros(device, shape),
            v: RawTensor::zeros(device, shape),
        }
    }
}

fn random_matrix(
    graph: &Graph,
    device: &Device,
    rng: &mut StdRng,
    rows: usize,
    cols: usize,
    scale: f32,
) -> Tensor<2> {
    let data: Vec<Vec<f32>> = (0..rows)
        .map(|_| (0..cols).map(|_| rng.random_range(-scale..scale)).collect())
        .collect();
    Tensor::new(graph, device, &data)
}

fn ones(graph: &Graph, device: &Device, len: usize) -> Tensor<1> {
    Tensor::new(graph, device, &vec![1.0; len])
}

fn zeros(graph: &Graph, device: &Device, len: usize) -> Tensor<1> {
    Tensor::new(graph, device, &vec![0.0; len])
}

fn adamw_update<const R: usize>(
    parameter: &Tensor<R>,
    moments: &mut AdamMoments<R>,
    gradients: &Gradients,
    step: usize,
    settings: AdamWSettings,
) -> Tensor<R> {
    let gradient = gradients.get(parameter).unwrap();
    let next_m = ((moments.m.clone() * settings.beta1) + (gradient.clone() * (1.0 - settings.beta1))).to_concrete();
    let next_v = ((moments.v.clone() * settings.beta2) + (gradient.sqr().to_concrete() * (1.0 - settings.beta2))).to_concrete();

    let bias_correction1 = 1.0 - settings.beta1.powi(step as i32);
    let bias_correction2 = 1.0 - settings.beta2.powi(step as i32);
    let m_hat = next_m.clone().div_scalar(bias_correction1).to_concrete();
    let v_hat = next_v.clone().div_scalar(bias_correction2).to_concrete();
    let adam_update = (m_hat / (v_hat.add_scalar(settings.adam_eps).sqrt().to_concrete())).to_concrete();
    let weight_decay = (parameter.raw().clone() * settings.weight_decay).to_concrete();
    let next_parameter = (parameter
        .raw()
        .clone()
        - ((adam_update + weight_decay).to_concrete() * settings.learning_rate))
        .to_concrete();

    moments.m = next_m;
    moments.v = next_v;
    Tensor::from_raw(&parameter.graph(), next_parameter)
}

fn tensor_len<const R: usize>(tensor: &Tensor<R>) -> usize {
    tensor.shape().iter().product()
}

async fn push_tensor_1d(tensors: &mut Vec<NamedTensor>, name: &str, tensor: &Tensor<1>) {
    let values = tensor.raw().clone().as_slice().await.unwrap().to_vec1();
    tensors.push(NamedTensor {
        name: name.to_string(),
        shape: tensor.shape().iter().map(|&dim| dim as u32).collect(),
        data: f32_bytes(&values),
    });
}

async fn push_tensor_2d(tensors: &mut Vec<NamedTensor>, name: &str, tensor: &Tensor<2>) {
    let values = tensor.raw().clone().as_slice().await.unwrap().to_vec2();
    let flat = values.into_iter().flatten().collect::<Vec<_>>();
    tensors.push(NamedTensor {
        name: name.to_string(),
        shape: tensor.shape().iter().map(|&dim| dim as u32).collect(),
        data: f32_bytes(&flat),
    });
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn is_attention_layer(index: usize, attention_period: usize) -> bool {
    (index + 1) % attention_period.max(1) == 0
}

impl ModelShape {
    fn head_dim(self) -> usize {
        self.n_embd / self.n_head.max(1)
    }
}
