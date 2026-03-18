use fusor::{
    Device, Tensor as RawTensor, ToVec1, ToVec2,
    autograd::{Gradients, Graph, Tensor},
};
use fusor_train::{AdamMoments, AdamWModel, AdamWSettings, adamw_update};
use rand::{Rng, rngs::StdRng};
use std::{io::Write, time::Instant};

use crate::config::RuntimeConfig;

pub struct NamedTensor {
    pub name: String,
    pub shape: Vec<u32>,
    pub values: Vec<f32>,
}

#[derive(Clone, Copy)]
struct ModelShape {
    block_size: usize,
    n_embd: usize,
    n_head: usize,
    n_ff: usize,
    conv_kernel_size: usize,
    eps: f32,
}

#[derive(Clone)]
pub struct NanoChatModel {
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

pub struct NanoChatAdamState {
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
    Conv(ConvMixer),
}

#[derive(Clone)]
struct CausalSelfAttention {
    c_attn_q: Tensor<2>,
    c_attn_k: Tensor<2>,
    c_attn_v: Tensor<2>,
    c_proj: Tensor<2>,
}

#[derive(Clone)]
struct ConvMixer {
    kernels: Vec<Tensor<2>>,
    bias: Tensor<1>,
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
    Conv(AdamConvState),
}

struct AdamAttentionState {
    c_attn_q: AdamMoments<2>,
    c_attn_k: AdamMoments<2>,
    c_attn_v: AdamMoments<2>,
    c_proj: AdamMoments<2>,
}

struct AdamConvState {
    kernels: Vec<AdamMoments<2>>,
    bias: AdamMoments<1>,
    out_proj: AdamMoments<2>,
}

struct AdamMlpState {
    c_fc: AdamMoments<2>,
    c_fc_bias: AdamMoments<1>,
    c_proj: AdamMoments<2>,
    c_proj_bias: AdamMoments<1>,
}

impl NanoChatModel {
    pub fn new(
        device: &Device,
        rng: &mut StdRng,
        vocab_size: usize,
        config: &RuntimeConfig,
    ) -> Self {
        let graph = Graph::new();
        let shape = ModelShape {
            block_size: config.block_size,
            n_embd: config.n_embd,
            n_head: config.n_head.max(1),
            n_ff: config.n_ff,
            conv_kernel_size: config.conv_kernel_size.max(1),
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
            wte: random_matrix(
                &graph,
                device,
                rng,
                vocab_size,
                shape.n_embd,
                config.init_scale,
            ),
            wpe: random_matrix(
                &graph,
                device,
                rng,
                shape.block_size,
                shape.n_embd,
                config.init_scale,
            ),
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
            lm_head: random_matrix(
                &graph,
                device,
                rng,
                shape.n_embd,
                vocab_size,
                config.init_scale,
            ),
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

    pub fn conv_kernel_size(&self) -> usize {
        self.shape.conv_kernel_size
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
        x.mat_mul(
            &self
                .lm_head
                .broadcast_as([batch_size, self.shape.n_embd, self.vocab_size]),
        )
    }

    pub async fn named_tensors(&self) -> Vec<NamedTensor> {
        let mut tensors = Vec::new();
        log_materialize_start("token_embd.weight", &self.wte.shape());
        push_tensor_2d(&mut tensors, "token_embd.weight", &self.wte).await;
        log_materialize_start("position_embd.weight", &self.wpe.shape());
        push_tensor_2d(&mut tensors, "position_embd.weight", &self.wpe).await;

        for (index, block) in self.blocks.iter().enumerate() {
            println!("materializing block {index} tensors...");
            let _ = std::io::stdout().flush();
            block.append_named_tensors(index, &mut tensors).await;
        }

        log_materialize_start("output_norm.weight", &self.ln_f_weight.shape());
        push_tensor_1d(&mut tensors, "output_norm.weight", &self.ln_f_weight).await;
        log_materialize_start("output_norm.bias", &self.ln_f_bias.shape());
        push_tensor_1d(&mut tensors, "output_norm.bias", &self.ln_f_bias).await;
        log_materialize_start("output.weight", &self.lm_head.shape());
        push_tensor_2d(&mut tensors, "output.weight", &self.lm_head).await;
        tensors
    }
}

impl AdamWModel for NanoChatModel {
    type State = NanoChatAdamState;

    fn adamw_state(device: &Device, model: &Self) -> Self::State {
        NanoChatAdamState {
            wte: AdamMoments::zeros_like(device, &model.wte),
            wpe: AdamMoments::zeros_like(device, &model.wpe),
            blocks: model
                .blocks
                .iter()
                .map(|block| AdamBlockState::new(device, block))
                .collect(),
            ln_f_weight: AdamMoments::zeros_like(device, &model.ln_f_weight),
            ln_f_bias: AdamMoments::zeros_like(device, &model.ln_f_bias),
            lm_head: AdamMoments::zeros_like(device, &model.lm_head),
        }
    }

    fn adamw_step(
        self,
        state: &mut Self::State,
        gradients: &Gradients,
        step: usize,
        settings: AdamWSettings,
    ) -> Self {
        let NanoChatModel {
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
        } = self;

        let blocks = blocks
            .into_iter()
            .zip(state.blocks.iter_mut())
            .map(|(block, block_state)| block_state.step(block, gradients, step, settings))
            .collect();

        NanoChatModel {
            graph,
            shape,
            attention_period,
            vocab_size,
            wte: adamw_update(&wte, &mut state.wte, gradients, step, settings),
            wpe: adamw_update(&wpe, &mut state.wpe, gradients, step, settings),
            blocks,
            ln_f_weight: adamw_update(
                &ln_f_weight,
                &mut state.ln_f_weight,
                gradients,
                step,
                settings,
            ),
            ln_f_bias: adamw_update(&ln_f_bias, &mut state.ln_f_bias, gradients, step, settings),
            lm_head: adamw_update(&lm_head, &mut state.lm_head, gradients, step, settings),
        }
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
                SequenceMixer::Attention(CausalSelfAttention::new(
                    graph, device, rng, shape, init_scale,
                ))
            } else {
                SequenceMixer::Conv(ConvMixer::new(graph, device, rng, shape, init_scale))
            },
            ln_2_weight: ones(graph, device, shape.n_embd),
            ln_2_bias: zeros(graph, device, shape.n_embd),
            mlp: Mlp::new(graph, device, rng, shape, init_scale),
        }
    }

    fn forward(
        &self,
        x: Tensor<3>,
        causal_mask: &Tensor<3>,
        batch_size: usize,
        shape: ModelShape,
    ) -> Tensor<3> {
        let attn_input = x.layer_norm(&self.ln_1_weight, Some(&self.ln_1_bias), shape.eps);
        let attn_output = self
            .mixer
            .forward(&attn_input, causal_mask, batch_size, shape);
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
        log_materialize_start(&format!("{prefix}.ln_1.weight"), &self.ln_1_weight.shape());
        push_tensor_1d(tensors, &format!("{prefix}.ln_1.weight"), &self.ln_1_weight).await;
        log_materialize_start(&format!("{prefix}.ln_1.bias"), &self.ln_1_bias.shape());
        push_tensor_1d(tensors, &format!("{prefix}.ln_1.bias"), &self.ln_1_bias).await;
        self.mixer.append_named_tensors(&prefix, tensors).await;
        log_materialize_start(&format!("{prefix}.ln_2.weight"), &self.ln_2_weight.shape());
        push_tensor_1d(tensors, &format!("{prefix}.ln_2.weight"), &self.ln_2_weight).await;
        log_materialize_start(&format!("{prefix}.ln_2.bias"), &self.ln_2_bias.shape());
        push_tensor_1d(tensors, &format!("{prefix}.ln_2.bias"), &self.ln_2_bias).await;
        self.mlp.append_named_tensors(&prefix, tensors).await;
    }
}

impl CausalSelfAttention {
    fn new(
        graph: &Graph,
        device: &Device,
        rng: &mut StdRng,
        shape: ModelShape,
        init_scale: f32,
    ) -> Self {
        Self {
            c_attn_q: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
            c_attn_k: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
            c_attn_v: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
            c_proj: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
        }
    }

    fn forward(
        &self,
        x: &Tensor<3>,
        causal_mask: &Tensor<3>,
        batch_size: usize,
        shape: ModelShape,
    ) -> Tensor<3> {
        let head_dim = shape.head_dim();
        let q = x.mat_mul(
            &self
                .c_attn_q
                .broadcast_as([batch_size, shape.n_embd, shape.n_embd]),
        );
        let k = x.mat_mul(
            &self
                .c_attn_k
                .broadcast_as([batch_size, shape.n_embd, shape.n_embd]),
        );
        let v = x.mat_mul(
            &self
                .c_attn_v
                .broadcast_as([batch_size, shape.n_embd, shape.n_embd]),
        );
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
        Tensor::cat(heads, 2).mat_mul(&self.c_proj.broadcast_as([
            batch_size,
            shape.n_embd,
            shape.n_embd,
        ]))
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.c_attn_q)
            + tensor_len(&self.c_attn_k)
            + tensor_len(&self.c_attn_v)
            + tensor_len(&self.c_proj)
    }

    async fn append_named_tensors(&self, prefix: &str, tensors: &mut Vec<NamedTensor>) {
        log_materialize_start(&format!("{prefix}.attn_q.weight"), &self.c_attn_q.shape());
        push_tensor_2d(tensors, &format!("{prefix}.attn_q.weight"), &self.c_attn_q).await;
        log_materialize_start(&format!("{prefix}.attn_k.weight"), &self.c_attn_k.shape());
        push_tensor_2d(tensors, &format!("{prefix}.attn_k.weight"), &self.c_attn_k).await;
        log_materialize_start(&format!("{prefix}.attn_v.weight"), &self.c_attn_v.shape());
        push_tensor_2d(tensors, &format!("{prefix}.attn_v.weight"), &self.c_attn_v).await;
        log_materialize_start(&format!("{prefix}.attn_proj.weight"), &self.c_proj.shape());
        push_tensor_2d(tensors, &format!("{prefix}.attn_proj.weight"), &self.c_proj).await;
    }
}

impl ConvMixer {
    fn new(
        graph: &Graph,
        device: &Device,
        rng: &mut StdRng,
        shape: ModelShape,
        init_scale: f32,
    ) -> Self {
        Self {
            kernels: (0..shape.conv_kernel_size)
                .map(|_| random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale))
                .collect(),
            bias: zeros(graph, device, shape.n_embd),
            out_proj: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
        }
    }

    fn forward(&self, x: &Tensor<3>, batch_size: usize, shape: ModelShape) -> Tensor<3> {
        let seq_len = x.shape()[1];
        let mut mixed = Tensor::zeros(&x.graph(), &x.device(), [batch_size, seq_len, shape.n_embd]);

        for (offset, kernel) in self.kernels.iter().enumerate() {
            let shifted = causal_shift_autograd(x, offset);
            let projected =
                shifted.mat_mul(&kernel.broadcast_as([batch_size, shape.n_embd, shape.n_embd]));
            mixed = mixed.add(&projected);
        }

        mixed
            .add(&self.bias.broadcast_as([batch_size, seq_len, shape.n_embd]))
            .relu()
            .mat_mul(
                &self
                    .out_proj
                    .broadcast_as([batch_size, shape.n_embd, shape.n_embd]),
            )
    }

    fn num_parameters(&self) -> usize {
        self.kernels.iter().map(tensor_len).sum::<usize>()
            + tensor_len(&self.bias)
            + tensor_len(&self.out_proj)
    }

    async fn append_named_tensors(&self, prefix: &str, tensors: &mut Vec<NamedTensor>) {
        for (index, kernel) in self.kernels.iter().enumerate() {
            log_materialize_start(
                &format!("{prefix}.conv_kernel.{index}.weight"),
                &kernel.shape(),
            );
            push_tensor_2d(
                tensors,
                &format!("{prefix}.conv_kernel.{index}.weight"),
                kernel,
            )
            .await;
        }
        log_materialize_start(&format!("{prefix}.conv_bias"), &self.bias.shape());
        push_tensor_1d(tensors, &format!("{prefix}.conv_bias"), &self.bias).await;
        log_materialize_start(
            &format!("{prefix}.conv_proj.weight"),
            &self.out_proj.shape(),
        );
        push_tensor_2d(
            tensors,
            &format!("{prefix}.conv_proj.weight"),
            &self.out_proj,
        )
        .await;
    }
}

impl SequenceMixer {
    fn forward(
        &self,
        x: &Tensor<3>,
        causal_mask: &Tensor<3>,
        batch_size: usize,
        shape: ModelShape,
    ) -> Tensor<3> {
        match self {
            SequenceMixer::Attention(attn) => attn.forward(x, causal_mask, batch_size, shape),
            SequenceMixer::Conv(conv) => conv.forward(x, batch_size, shape),
        }
    }

    fn num_parameters(&self) -> usize {
        match self {
            SequenceMixer::Attention(attn) => attn.num_parameters(),
            SequenceMixer::Conv(conv) => conv.num_parameters(),
        }
    }

    async fn append_named_tensors(&self, prefix: &str, tensors: &mut Vec<NamedTensor>) {
        match self {
            SequenceMixer::Attention(attn) => attn.append_named_tensors(prefix, tensors).await,
            SequenceMixer::Conv(conv) => conv.append_named_tensors(prefix, tensors).await,
        }
    }
}

impl Mlp {
    fn new(
        graph: &Graph,
        device: &Device,
        rng: &mut StdRng,
        shape: ModelShape,
        init_scale: f32,
    ) -> Self {
        Self {
            c_fc: random_matrix(graph, device, rng, shape.n_embd, shape.n_ff, init_scale),
            c_fc_bias: zeros(graph, device, shape.n_ff),
            c_proj: random_matrix(graph, device, rng, shape.n_ff, shape.n_embd, init_scale),
            c_proj_bias: zeros(graph, device, shape.n_embd),
        }
    }

    fn forward(&self, x: &Tensor<3>, batch_size: usize, shape: ModelShape) -> Tensor<3> {
        let hidden = x
            .mat_mul(
                &self
                    .c_fc
                    .broadcast_as([batch_size, shape.n_embd, shape.n_ff]),
            )
            .add(
                &self
                    .c_fc_bias
                    .broadcast_as([batch_size, shape.block_size, shape.n_ff]),
            )
            .relu();

        hidden
            .mat_mul(
                &self
                    .c_proj
                    .broadcast_as([batch_size, shape.n_ff, shape.n_embd]),
            )
            .add(
                &self
                    .c_proj_bias
                    .broadcast_as([batch_size, shape.block_size, shape.n_embd]),
            )
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.c_fc)
            + tensor_len(&self.c_fc_bias)
            + tensor_len(&self.c_proj)
            + tensor_len(&self.c_proj_bias)
    }

    async fn append_named_tensors(&self, prefix: &str, tensors: &mut Vec<NamedTensor>) {
        log_materialize_start(&format!("{prefix}.mlp_fc.weight"), &self.c_fc.shape());
        push_tensor_2d(tensors, &format!("{prefix}.mlp_fc.weight"), &self.c_fc).await;
        log_materialize_start(&format!("{prefix}.mlp_fc.bias"), &self.c_fc_bias.shape());
        push_tensor_1d(tensors, &format!("{prefix}.mlp_fc.bias"), &self.c_fc_bias).await;
        log_materialize_start(&format!("{prefix}.mlp_proj.weight"), &self.c_proj.shape());
        push_tensor_2d(tensors, &format!("{prefix}.mlp_proj.weight"), &self.c_proj).await;
        log_materialize_start(
            &format!("{prefix}.mlp_proj.bias"),
            &self.c_proj_bias.shape(),
        );
        push_tensor_1d(
            tensors,
            &format!("{prefix}.mlp_proj.bias"),
            &self.c_proj_bias,
        )
        .await;
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

    fn step(
        &mut self,
        block: TransformerBlock,
        gradients: &Gradients,
        step: usize,
        settings: AdamWSettings,
    ) -> TransformerBlock {
        TransformerBlock {
            ln_1_weight: adamw_update(
                &block.ln_1_weight,
                &mut self.ln_1_weight,
                gradients,
                step,
                settings,
            ),
            ln_1_bias: adamw_update(
                &block.ln_1_bias,
                &mut self.ln_1_bias,
                gradients,
                step,
                settings,
            ),
            mixer: self.mixer.step(block.mixer, gradients, step, settings),
            ln_2_weight: adamw_update(
                &block.ln_2_weight,
                &mut self.ln_2_weight,
                gradients,
                step,
                settings,
            ),
            ln_2_bias: adamw_update(
                &block.ln_2_bias,
                &mut self.ln_2_bias,
                gradients,
                step,
                settings,
            ),
            mlp: self.mlp.step(block.mlp, gradients, step, settings),
        }
    }
}

impl AdamMixerState {
    fn new(device: &Device, mixer: &SequenceMixer) -> Self {
        match mixer {
            SequenceMixer::Attention(attn) => {
                Self::Attention(AdamAttentionState::new(device, attn))
            }
            SequenceMixer::Conv(conv) => Self::Conv(AdamConvState::new(device, conv)),
        }
    }

    fn step(
        &mut self,
        mixer: SequenceMixer,
        gradients: &Gradients,
        step: usize,
        settings: AdamWSettings,
    ) -> SequenceMixer {
        match (self, mixer) {
            (AdamMixerState::Attention(state), SequenceMixer::Attention(attn)) => {
                SequenceMixer::Attention(state.step(attn, gradients, step, settings))
            }
            (AdamMixerState::Conv(state), SequenceMixer::Conv(conv)) => {
                SequenceMixer::Conv(state.step(conv, gradients, step, settings))
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

    fn step(
        &mut self,
        attn: CausalSelfAttention,
        gradients: &Gradients,
        step: usize,
        settings: AdamWSettings,
    ) -> CausalSelfAttention {
        CausalSelfAttention {
            c_attn_q: adamw_update(
                &attn.c_attn_q,
                &mut self.c_attn_q,
                gradients,
                step,
                settings,
            ),
            c_attn_k: adamw_update(
                &attn.c_attn_k,
                &mut self.c_attn_k,
                gradients,
                step,
                settings,
            ),
            c_attn_v: adamw_update(
                &attn.c_attn_v,
                &mut self.c_attn_v,
                gradients,
                step,
                settings,
            ),
            c_proj: adamw_update(&attn.c_proj, &mut self.c_proj, gradients, step, settings),
        }
    }
}

impl AdamConvState {
    fn new(device: &Device, conv: &ConvMixer) -> Self {
        Self {
            kernels: conv
                .kernels
                .iter()
                .map(|kernel| AdamMoments::zeros_like(device, kernel))
                .collect(),
            bias: AdamMoments::zeros_like(device, &conv.bias),
            out_proj: AdamMoments::zeros_like(device, &conv.out_proj),
        }
    }

    fn step(
        &mut self,
        conv: ConvMixer,
        gradients: &Gradients,
        step: usize,
        settings: AdamWSettings,
    ) -> ConvMixer {
        let kernels = conv
            .kernels
            .into_iter()
            .zip(self.kernels.iter_mut())
            .map(|(kernel, state)| adamw_update(&kernel, state, gradients, step, settings))
            .collect();
        ConvMixer {
            kernels,
            bias: adamw_update(&conv.bias, &mut self.bias, gradients, step, settings),
            out_proj: adamw_update(
                &conv.out_proj,
                &mut self.out_proj,
                gradients,
                step,
                settings,
            ),
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

    fn step(
        &mut self,
        mlp: Mlp,
        gradients: &Gradients,
        step: usize,
        settings: AdamWSettings,
    ) -> Mlp {
        Mlp {
            c_fc: adamw_update(&mlp.c_fc, &mut self.c_fc, gradients, step, settings),
            c_fc_bias: adamw_update(
                &mlp.c_fc_bias,
                &mut self.c_fc_bias,
                gradients,
                step,
                settings,
            ),
            c_proj: adamw_update(&mlp.c_proj, &mut self.c_proj, gradients, step, settings),
            c_proj_bias: adamw_update(
                &mlp.c_proj_bias,
                &mut self.c_proj_bias,
                gradients,
                step,
                settings,
            ),
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

fn tensor_len<const R: usize>(tensor: &Tensor<R>) -> usize {
    tensor.shape().iter().product()
}

async fn push_tensor_1d(tensors: &mut Vec<NamedTensor>, name: &str, tensor: &Tensor<1>) {
    let start = Instant::now();
    let values = tensor.raw().clone().as_slice().await.unwrap().to_vec1();
    println!(
        "materialized tensor {name} shape={:?} elems={} bytes={} in {:.2?}",
        tensor.shape(),
        values.len(),
        values.len() * std::mem::size_of::<f32>(),
        start.elapsed(),
    );
    tensors.push(NamedTensor {
        name: name.to_string(),
        shape: tensor.shape().iter().map(|&dim| dim as u32).collect(),
        values,
    });
}

async fn push_tensor_2d(tensors: &mut Vec<NamedTensor>, name: &str, tensor: &Tensor<2>) {
    let start = Instant::now();
    let values = tensor.raw().clone().as_slice().await.unwrap().to_vec2();
    let flat = values.into_iter().flatten().collect::<Vec<_>>();
    println!(
        "materialized tensor {name} shape={:?} elems={} bytes={} in {:.2?}",
        tensor.shape(),
        flat.len(),
        flat.len() * std::mem::size_of::<f32>(),
        start.elapsed(),
    );
    tensors.push(NamedTensor {
        name: name.to_string(),
        shape: tensor.shape().iter().map(|&dim| dim as u32).collect(),
        values: flat,
    });
}

fn log_materialize_start<const R: usize>(name: &str, shape: &[usize; R]) {
    let elements: usize = shape.iter().product();
    println!(
        "starting materialize {name} shape={shape:?} elems={} bytes={}",
        elements,
        elements * std::mem::size_of::<f32>(),
    );
    let _ = std::io::stdout().flush();
}

fn is_attention_layer(index: usize, attention_period: usize) -> bool {
    (index + 1) % attention_period.max(1) == 0
}

fn causal_shift_autograd(x: &Tensor<3>, offset: usize) -> Tensor<3> {
    if offset == 0 {
        return x.clone();
    }

    let [batch_size, seq_len, n_embd] = x.shape();
    if offset >= seq_len {
        return Tensor::zeros(&x.graph(), &x.device(), [batch_size, seq_len, n_embd]);
    }

    let prefix = Tensor::zeros(&x.graph(), &x.device(), [batch_size, offset, n_embd]);
    let shifted = x.slice([0..batch_size, 0..seq_len - offset, 0..n_embd]);
    Tensor::cat(vec![prefix, shifted], 1)
}

impl ModelShape {
    fn head_dim(self) -> usize {
        self.n_embd / self.n_head.max(1)
    }
}
