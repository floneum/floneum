use fusor::{
    Device, Tensor as RawTensor, ToVec1, ToVec2,
    autograd::{Gradients, Graph, Tensor},
    base_inverse_frequency,
};
use fusor_train::{AdamMoments, AdamWModel, AdamWSettings, adamw_update};
use rand::{Rng, rngs::StdRng};
use std::{io::Write, time::Instant};

use crate::{
    config::RuntimeConfig,
    data::{
        ACTION_DIRECTION_COUNT, ACTION_MODE_COUNT, CanvasStateSpec, StrokeTokenizer,
        canvas_state_spec,
    },
};

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
    n_kv_head: usize,
    n_ff: usize,
    conv_kernel_size: usize,
    eps: f32,
}

#[derive(Clone)]
struct RotaryEmbeddings {
    cos: Tensor<2>,
    sin: Tensor<2>,
}

#[derive(Clone)]
struct CanvasStateEmbeddings {
    spec: CanvasStateSpec,
    cursor_x: Tensor<2>,
    cursor_y: Tensor<2>,
    pen_state: Tensor<2>,
}

#[derive(Clone)]
struct OutputHead {
    weight: Tensor<2>,
    bias: Tensor<1>,
}

pub struct ActionLogits {
    pub mode: Tensor<3>,
    pub direction: Tensor<3>,
    pub length_ordinal: Tensor<3>,
    pub length_scalar: Tensor<3>,
}

#[derive(Clone)]
pub struct NanoChatModel {
    graph: Graph,
    shape: ModelShape,
    attention_period: usize,
    vocab_size: usize,
    max_count: usize,
    use_rope: bool,
    rope_theta: f32,
    use_extra_norms: bool,
    wte: Tensor<2>,
    wpe: Option<Tensor<2>>,
    canvas_state: Option<CanvasStateEmbeddings>,
    rotary: Option<RotaryEmbeddings>,
    ln_in_weight: Tensor<1>,
    ln_in_bias: Tensor<1>,
    blocks: Vec<TransformerBlock>,
    ln_f_weight: Tensor<1>,
    ln_f_bias: Tensor<1>,
    mode_head: OutputHead,
    direction_head: OutputHead,
    length_ordinal_head: OutputHead,
    length_scalar_head: OutputHead,
}

pub struct NanoChatAdamState {
    wte: AdamMoments<2>,
    wpe: Option<AdamMoments<2>>,
    canvas_state: Option<CanvasStateAdamState>,
    ln_in_weight: AdamMoments<1>,
    ln_in_bias: AdamMoments<1>,
    blocks: Vec<AdamBlockState>,
    ln_f_weight: AdamMoments<1>,
    ln_f_bias: AdamMoments<1>,
    mode_head: AdamOutputHeadState,
    direction_head: AdamOutputHeadState,
    length_ordinal_head: AdamOutputHeadState,
    length_scalar_head: AdamOutputHeadState,
}

#[derive(Clone)]
struct TransformerBlock {
    ln_1_weight: Tensor<1>,
    ln_1_bias: Tensor<1>,
    mixer: SequenceMixer,
    ln_attn_out_weight: Tensor<1>,
    ln_attn_out_bias: Tensor<1>,
    ln_2_weight: Tensor<1>,
    ln_2_bias: Tensor<1>,
    mlp: Mlp,
    ln_mlp_out_weight: Tensor<1>,
    ln_mlp_out_bias: Tensor<1>,
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
    ln_attn_out_weight: AdamMoments<1>,
    ln_attn_out_bias: AdamMoments<1>,
    ln_2_weight: AdamMoments<1>,
    ln_2_bias: AdamMoments<1>,
    mlp: AdamMlpState,
    ln_mlp_out_weight: AdamMoments<1>,
    ln_mlp_out_bias: AdamMoments<1>,
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

struct CanvasStateAdamState {
    cursor_x: AdamMoments<2>,
    cursor_y: AdamMoments<2>,
    pen_state: AdamMoments<2>,
}

struct AdamOutputHeadState {
    weight: AdamMoments<2>,
    bias: AdamMoments<1>,
}

impl NanoChatModel {
    pub fn new(
        device: &Device,
        rng: &mut StdRng,
        tokenizer: &StrokeTokenizer,
        config: &RuntimeConfig,
    ) -> Self {
        let vocab_size = tokenizer.vocab_size();
        let graph = Graph::new();
        let shape = ModelShape {
            block_size: config.block_size,
            n_embd: config.n_embd,
            n_head: config.n_head.max(1),
            n_kv_head: config.n_kv_head.max(1),
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
        assert_eq!(
            shape.n_head % shape.n_kv_head,
            0,
            "NANOCHAT_N_HEAD ({}) must be divisible by NANOCHAT_N_KV_HEAD ({})",
            shape.n_head,
            shape.n_kv_head
        );
        if config.use_rope {
            assert_eq!(
                shape.head_dim() % 2,
                0,
                "RoPE requires an even head dimension, got {}",
                shape.head_dim()
            );
        }

        Self {
            graph: graph.clone(),
            shape,
            attention_period: config.attention_period.max(1),
            vocab_size,
            max_count: tokenizer.max_count(),
            use_rope: config.use_rope,
            rope_theta: config.rope_theta,
            use_extra_norms: config.use_extra_norms,
            wte: random_matrix(
                &graph,
                device,
                rng,
                vocab_size,
                shape.n_embd,
                config.init_scale,
            ),
            wpe: (!config.use_rope).then(|| {
                random_matrix(
                    &graph,
                    device,
                    rng,
                    shape.block_size,
                    shape.n_embd,
                    config.init_scale,
                )
            }),
            canvas_state: config.use_canvas_state_embeddings.then(|| {
                CanvasStateEmbeddings::new(
                    &graph,
                    device,
                    rng,
                    canvas_state_spec(tokenizer, config.block_size),
                    shape.n_embd,
                    config.init_scale,
                )
            }),
            rotary: config
                .use_rope
                .then(|| RotaryEmbeddings::new(&graph, device, shape, config.rope_theta)),
            ln_in_weight: ones(&graph, device, shape.n_embd),
            ln_in_bias: zeros(&graph, device, shape.n_embd),
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
            mode_head: OutputHead::new(
                &graph,
                device,
                rng,
                shape.n_embd,
                ACTION_MODE_COUNT,
                config.init_scale,
            ),
            direction_head: OutputHead::new(
                &graph,
                device,
                rng,
                shape.n_embd,
                ACTION_DIRECTION_COUNT,
                config.init_scale,
            ),
            length_ordinal_head: OutputHead::new(
                &graph,
                device,
                rng,
                shape.n_embd,
                tokenizer.max_count().saturating_sub(1).max(1),
                config.init_scale,
            ),
            length_scalar_head: OutputHead::new(
                &graph,
                device,
                rng,
                shape.n_embd,
                1,
                config.init_scale,
            ),
        }
    }

    pub fn num_parameters(&self) -> usize {
        tensor_len(&self.wte)
            + self.wpe.as_ref().map_or(0, tensor_len)
            + self
                .canvas_state
                .as_ref()
                .map_or(0, CanvasStateEmbeddings::num_parameters)
            + tensor_len(&self.ln_in_weight)
            + tensor_len(&self.ln_in_bias)
            + self
                .blocks
                .iter()
                .map(TransformerBlock::num_parameters)
                .sum::<usize>()
            + tensor_len(&self.ln_f_weight)
            + tensor_len(&self.ln_f_bias)
            + self.mode_head.num_parameters()
            + self.direction_head.num_parameters()
            + self.length_ordinal_head.num_parameters()
            + self.length_scalar_head.num_parameters()
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    fn into_graph(self, graph: Graph) -> Self {
        let NanoChatModel {
            graph: _,
            shape,
            attention_period,
            vocab_size,
            max_count,
            use_rope,
            rope_theta,
            use_extra_norms,
            wte,
            wpe,
            canvas_state,
            rotary,
            ln_in_weight,
            ln_in_bias,
            blocks,
            ln_f_weight,
            ln_f_bias,
            mode_head,
            direction_head,
            length_ordinal_head,
            length_scalar_head,
        } = self;

        Self {
            graph: graph.clone(),
            shape,
            attention_period,
            vocab_size,
            max_count,
            use_rope,
            rope_theta,
            use_extra_norms,
            wte: regraph_tensor(&graph, wte),
            wpe: wpe.map(|tensor| regraph_tensor(&graph, tensor)),
            canvas_state: canvas_state.map(|state| state.into_graph(&graph)),
            rotary: rotary.map(|cache| cache.into_graph(&graph)),
            ln_in_weight: regraph_tensor(&graph, ln_in_weight),
            ln_in_bias: regraph_tensor(&graph, ln_in_bias),
            blocks: blocks
                .into_iter()
                .map(|block| block.into_graph(&graph))
                .collect(),
            ln_f_weight: regraph_tensor(&graph, ln_f_weight),
            ln_f_bias: regraph_tensor(&graph, ln_f_bias),
            mode_head: mode_head.into_graph(&graph),
            direction_head: direction_head.into_graph(&graph),
            length_ordinal_head: length_ordinal_head.into_graph(&graph),
            length_scalar_head: length_scalar_head.into_graph(&graph),
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn max_count(&self) -> usize {
        self.max_count
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

    pub fn n_kv_head(&self) -> usize {
        self.shape.n_kv_head
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

    pub fn use_rope(&self) -> bool {
        self.use_rope
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    pub fn use_extra_norms(&self) -> bool {
        self.use_extra_norms
    }

    pub fn canvas_state_spec(&self) -> Option<CanvasStateSpec> {
        self.canvas_state.as_ref().map(|state| state.spec)
    }

    pub fn forward(
        &self,
        token_inputs: &RawTensor<2, u32>,
        position_inputs: &RawTensor<2, u32>,
        cursor_x_inputs: &RawTensor<2, u32>,
        cursor_y_inputs: &RawTensor<2, u32>,
        pen_state_inputs: &RawTensor<2, u32>,
        causal_mask: &Tensor<3>,
    ) -> ActionLogits {
        let batch_size = token_inputs.shape()[0];
        let token_embeddings = self.wte.embedding(token_inputs);
        let mut x = token_embeddings;
        if let Some(wpe) = &self.wpe {
            let position_embeddings = wpe.embedding(position_inputs);
            x = x.add(&position_embeddings);
        }
        if let Some(canvas_state) = &self.canvas_state {
            x = x
                .add(&canvas_state.cursor_x.embedding(cursor_x_inputs))
                .add(&canvas_state.cursor_y.embedding(cursor_y_inputs))
                .add(&canvas_state.pen_state.embedding(pen_state_inputs));
        }
        if self.use_extra_norms {
            x = x.layer_norm(&self.ln_in_weight, Some(&self.ln_in_bias), self.shape.eps);
        }

        for block in &self.blocks {
            x = block.forward(
                x,
                causal_mask,
                batch_size,
                self.shape,
                self.rotary.as_ref(),
                self.use_extra_norms,
            );
        }

        let x = x.layer_norm(&self.ln_f_weight, Some(&self.ln_f_bias), self.shape.eps);
        ActionLogits {
            mode: self.mode_head.project(&x, batch_size, self.shape.n_embd),
            direction: self
                .direction_head
                .project(&x, batch_size, self.shape.n_embd),
            length_ordinal: self
                .length_ordinal_head
                .project(&x, batch_size, self.shape.n_embd),
            length_scalar: self
                .length_scalar_head
                .project(&x, batch_size, self.shape.n_embd),
        }
    }

    pub async fn named_tensors(&self) -> Vec<NamedTensor> {
        let mut tensors = Vec::new();
        log_materialize_start("token_embd.weight", &self.wte.shape());
        push_tensor_2d(&mut tensors, "token_embd.weight", &self.wte).await;
        if let Some(wpe) = &self.wpe {
            log_materialize_start("position_embd.weight", &wpe.shape());
            push_tensor_2d(&mut tensors, "position_embd.weight", wpe).await;
        }
        if let Some(canvas_state) = &self.canvas_state {
            canvas_state.append_named_tensors(&mut tensors).await;
        }
        log_materialize_start("input_norm.weight", &self.ln_in_weight.shape());
        push_tensor_1d(&mut tensors, "input_norm.weight", &self.ln_in_weight).await;
        log_materialize_start("input_norm.bias", &self.ln_in_bias.shape());
        push_tensor_1d(&mut tensors, "input_norm.bias", &self.ln_in_bias).await;

        for (index, block) in self.blocks.iter().enumerate() {
            println!("materializing block {index} tensors...");
            let _ = std::io::stdout().flush();
            block.append_named_tensors(index, &mut tensors).await;
        }

        log_materialize_start("output_norm.weight", &self.ln_f_weight.shape());
        push_tensor_1d(&mut tensors, "output_norm.weight", &self.ln_f_weight).await;
        log_materialize_start("output_norm.bias", &self.ln_f_bias.shape());
        push_tensor_1d(&mut tensors, "output_norm.bias", &self.ln_f_bias).await;
        self.mode_head
            .append_named_tensors("output_mode", &mut tensors)
            .await;
        self.direction_head
            .append_named_tensors("output_direction", &mut tensors)
            .await;
        self.length_ordinal_head
            .append_named_tensors("output_length_ordinal", &mut tensors)
            .await;
        self.length_scalar_head
            .append_named_tensors("output_length_scalar", &mut tensors)
            .await;
        tensors
    }
}

impl AdamWModel for NanoChatModel {
    type State = NanoChatAdamState;

    fn adamw_state(device: &Device, model: &Self) -> Self::State {
        NanoChatAdamState {
            wte: AdamMoments::zeros_like(device, &model.wte),
            wpe: model
                .wpe
                .as_ref()
                .map(|wpe| AdamMoments::zeros_like(device, wpe)),
            canvas_state: model
                .canvas_state
                .as_ref()
                .map(|state| CanvasStateAdamState::new(device, state)),
            ln_in_weight: AdamMoments::zeros_like(device, &model.ln_in_weight),
            ln_in_bias: AdamMoments::zeros_like(device, &model.ln_in_bias),
            blocks: model
                .blocks
                .iter()
                .map(|block| AdamBlockState::new(device, block))
                .collect(),
            ln_f_weight: AdamMoments::zeros_like(device, &model.ln_f_weight),
            ln_f_bias: AdamMoments::zeros_like(device, &model.ln_f_bias),
            mode_head: AdamOutputHeadState::new(device, &model.mode_head),
            direction_head: AdamOutputHeadState::new(device, &model.direction_head),
            length_ordinal_head: AdamOutputHeadState::new(device, &model.length_ordinal_head),
            length_scalar_head: AdamOutputHeadState::new(device, &model.length_scalar_head),
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
            max_count,
            use_rope,
            rope_theta,
            use_extra_norms,
            wte,
            wpe,
            canvas_state,
            rotary,
            ln_in_weight,
            ln_in_bias,
            blocks,
            ln_f_weight,
            ln_f_bias,
            mode_head,
            direction_head,
            length_ordinal_head,
            length_scalar_head,
        } = self;

        let blocks = blocks
            .into_iter()
            .zip(state.blocks.iter_mut())
            .map(|(block, block_state)| block_state.step(block, gradients, step, settings))
            .collect();

        let updated = NanoChatModel {
            graph,
            shape,
            attention_period,
            vocab_size,
            max_count,
            use_rope,
            rope_theta,
            use_extra_norms,
            wte: adamw_update(&wte, &mut state.wte, gradients, step, settings),
            wpe: adamw_update_optional(wpe, &mut state.wpe, gradients, step, settings),
            canvas_state: match (canvas_state, state.canvas_state.as_mut()) {
                (Some(canvas_state), Some(state)) => {
                    Some(state.step(canvas_state, gradients, step, settings))
                }
                (None, None) => None,
                _ => unreachable!("optimizer state does not match canvas state embeddings"),
            },
            rotary,
            ln_in_weight: adamw_update(
                &ln_in_weight,
                &mut state.ln_in_weight,
                gradients,
                step,
                settings,
            ),
            ln_in_bias: adamw_update(
                &ln_in_bias,
                &mut state.ln_in_bias,
                gradients,
                step,
                settings,
            ),
            blocks,
            ln_f_weight: adamw_update(
                &ln_f_weight,
                &mut state.ln_f_weight,
                gradients,
                step,
                settings,
            ),
            ln_f_bias: adamw_update(&ln_f_bias, &mut state.ln_f_bias, gradients, step, settings),
            mode_head: state.mode_head.step(mode_head, gradients, step, settings),
            direction_head: state
                .direction_head
                .step(direction_head, gradients, step, settings),
            length_ordinal_head: state.length_ordinal_head.step(
                length_ordinal_head,
                gradients,
                step,
                settings,
            ),
            length_scalar_head: state.length_scalar_head.step(
                length_scalar_head,
                gradients,
                step,
                settings,
            ),
        };

        updated.into_graph(Graph::new())
    }
}

impl CanvasStateEmbeddings {
    fn new(
        graph: &Graph,
        device: &Device,
        rng: &mut StdRng,
        spec: CanvasStateSpec,
        n_embd: usize,
        init_scale: f32,
    ) -> Self {
        Self {
            spec,
            cursor_x: random_matrix(
                graph,
                device,
                rng,
                spec.coordinate_vocab_size,
                n_embd,
                init_scale,
            ),
            cursor_y: random_matrix(
                graph,
                device,
                rng,
                spec.coordinate_vocab_size,
                n_embd,
                init_scale,
            ),
            pen_state: random_matrix(graph, device, rng, 2, n_embd, init_scale),
        }
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.cursor_x) + tensor_len(&self.cursor_y) + tensor_len(&self.pen_state)
    }

    async fn append_named_tensors(&self, tensors: &mut Vec<NamedTensor>) {
        log_materialize_start("cursor_x_embd.weight", &self.cursor_x.shape());
        push_tensor_2d(tensors, "cursor_x_embd.weight", &self.cursor_x).await;
        log_materialize_start("cursor_y_embd.weight", &self.cursor_y.shape());
        push_tensor_2d(tensors, "cursor_y_embd.weight", &self.cursor_y).await;
        log_materialize_start("pen_state_embd.weight", &self.pen_state.shape());
        push_tensor_2d(tensors, "pen_state_embd.weight", &self.pen_state).await;
    }

    fn into_graph(self, graph: &Graph) -> Self {
        Self {
            spec: self.spec,
            cursor_x: regraph_tensor(graph, self.cursor_x),
            cursor_y: regraph_tensor(graph, self.cursor_y),
            pen_state: regraph_tensor(graph, self.pen_state),
        }
    }
}

impl CanvasStateAdamState {
    fn new(device: &Device, state: &CanvasStateEmbeddings) -> Self {
        Self {
            cursor_x: AdamMoments::zeros_like(device, &state.cursor_x),
            cursor_y: AdamMoments::zeros_like(device, &state.cursor_y),
            pen_state: AdamMoments::zeros_like(device, &state.pen_state),
        }
    }

    fn step(
        &mut self,
        state: CanvasStateEmbeddings,
        gradients: &Gradients,
        step: usize,
        settings: AdamWSettings,
    ) -> CanvasStateEmbeddings {
        CanvasStateEmbeddings {
            spec: state.spec,
            cursor_x: adamw_update(
                &state.cursor_x,
                &mut self.cursor_x,
                gradients,
                step,
                settings,
            ),
            cursor_y: adamw_update(
                &state.cursor_y,
                &mut self.cursor_y,
                gradients,
                step,
                settings,
            ),
            pen_state: adamw_update(
                &state.pen_state,
                &mut self.pen_state,
                gradients,
                step,
                settings,
            ),
        }
    }
}

impl OutputHead {
    fn new(
        graph: &Graph,
        device: &Device,
        rng: &mut StdRng,
        input_dim: usize,
        output_dim: usize,
        init_scale: f32,
    ) -> Self {
        Self {
            weight: random_matrix(graph, device, rng, input_dim, output_dim, init_scale),
            bias: zeros(graph, device, output_dim),
        }
    }

    fn project(&self, x: &Tensor<3>, batch_size: usize, input_dim: usize) -> Tensor<3> {
        let seq_len = x.shape()[1];
        x.mat_mul(
            &self
                .weight
                .broadcast_as([batch_size, input_dim, self.bias.shape()[0]]),
        )
        .add(
            &self
                .bias
                .broadcast_as([batch_size, seq_len, self.bias.shape()[0]]),
        )
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.weight) + tensor_len(&self.bias)
    }

    async fn append_named_tensors(&self, prefix: &str, tensors: &mut Vec<NamedTensor>) {
        log_materialize_start(&format!("{prefix}.weight"), &self.weight.shape());
        push_tensor_2d(tensors, &format!("{prefix}.weight"), &self.weight).await;
        log_materialize_start(&format!("{prefix}.bias"), &self.bias.shape());
        push_tensor_1d(tensors, &format!("{prefix}.bias"), &self.bias).await;
    }

    fn into_graph(self, graph: &Graph) -> Self {
        Self {
            weight: regraph_tensor(graph, self.weight),
            bias: regraph_tensor(graph, self.bias),
        }
    }
}

impl AdamOutputHeadState {
    fn new(device: &Device, head: &OutputHead) -> Self {
        Self {
            weight: AdamMoments::zeros_like(device, &head.weight),
            bias: AdamMoments::zeros_like(device, &head.bias),
        }
    }

    fn step(
        &mut self,
        head: OutputHead,
        gradients: &Gradients,
        step: usize,
        settings: AdamWSettings,
    ) -> OutputHead {
        OutputHead {
            weight: adamw_update(&head.weight, &mut self.weight, gradients, step, settings),
            bias: adamw_update(&head.bias, &mut self.bias, gradients, step, settings),
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
            ln_attn_out_weight: ones(graph, device, shape.n_embd),
            ln_attn_out_bias: zeros(graph, device, shape.n_embd),
            ln_2_weight: ones(graph, device, shape.n_embd),
            ln_2_bias: zeros(graph, device, shape.n_embd),
            mlp: Mlp::new(graph, device, rng, shape, init_scale),
            ln_mlp_out_weight: ones(graph, device, shape.n_embd),
            ln_mlp_out_bias: zeros(graph, device, shape.n_embd),
        }
    }

    fn forward(
        &self,
        x: Tensor<3>,
        causal_mask: &Tensor<3>,
        batch_size: usize,
        shape: ModelShape,
        rotary: Option<&RotaryEmbeddings>,
        use_extra_norms: bool,
    ) -> Tensor<3> {
        let attn_input = x.layer_norm(&self.ln_1_weight, Some(&self.ln_1_bias), shape.eps);
        let mut attn_output =
            self.mixer
                .forward(&attn_input, causal_mask, batch_size, shape, rotary);
        if use_extra_norms {
            attn_output = attn_output.layer_norm(
                &self.ln_attn_out_weight,
                Some(&self.ln_attn_out_bias),
                shape.eps,
            );
        }
        let x = x.add(&attn_output);
        let mlp_input = x.layer_norm(&self.ln_2_weight, Some(&self.ln_2_bias), shape.eps);
        let mut mlp_output = self.mlp.forward(&mlp_input, batch_size, shape);
        if use_extra_norms {
            mlp_output = mlp_output.layer_norm(
                &self.ln_mlp_out_weight,
                Some(&self.ln_mlp_out_bias),
                shape.eps,
            );
        }
        x.add(&mlp_output)
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.ln_1_weight)
            + tensor_len(&self.ln_1_bias)
            + self.mixer.num_parameters()
            + tensor_len(&self.ln_attn_out_weight)
            + tensor_len(&self.ln_attn_out_bias)
            + tensor_len(&self.ln_2_weight)
            + tensor_len(&self.ln_2_bias)
            + self.mlp.num_parameters()
            + tensor_len(&self.ln_mlp_out_weight)
            + tensor_len(&self.ln_mlp_out_bias)
    }

    async fn append_named_tensors(&self, index: usize, tensors: &mut Vec<NamedTensor>) {
        let prefix = format!("blk.{index}");
        log_materialize_start(&format!("{prefix}.ln_1.weight"), &self.ln_1_weight.shape());
        push_tensor_1d(tensors, &format!("{prefix}.ln_1.weight"), &self.ln_1_weight).await;
        log_materialize_start(&format!("{prefix}.ln_1.bias"), &self.ln_1_bias.shape());
        push_tensor_1d(tensors, &format!("{prefix}.ln_1.bias"), &self.ln_1_bias).await;
        self.mixer.append_named_tensors(&prefix, tensors).await;
        log_materialize_start(
            &format!("{prefix}.attn_out_norm.weight"),
            &self.ln_attn_out_weight.shape(),
        );
        push_tensor_1d(
            tensors,
            &format!("{prefix}.attn_out_norm.weight"),
            &self.ln_attn_out_weight,
        )
        .await;
        log_materialize_start(
            &format!("{prefix}.attn_out_norm.bias"),
            &self.ln_attn_out_bias.shape(),
        );
        push_tensor_1d(
            tensors,
            &format!("{prefix}.attn_out_norm.bias"),
            &self.ln_attn_out_bias,
        )
        .await;
        log_materialize_start(&format!("{prefix}.ln_2.weight"), &self.ln_2_weight.shape());
        push_tensor_1d(tensors, &format!("{prefix}.ln_2.weight"), &self.ln_2_weight).await;
        log_materialize_start(&format!("{prefix}.ln_2.bias"), &self.ln_2_bias.shape());
        push_tensor_1d(tensors, &format!("{prefix}.ln_2.bias"), &self.ln_2_bias).await;
        self.mlp.append_named_tensors(&prefix, tensors).await;
        log_materialize_start(
            &format!("{prefix}.mlp_out_norm.weight"),
            &self.ln_mlp_out_weight.shape(),
        );
        push_tensor_1d(
            tensors,
            &format!("{prefix}.mlp_out_norm.weight"),
            &self.ln_mlp_out_weight,
        )
        .await;
        log_materialize_start(
            &format!("{prefix}.mlp_out_norm.bias"),
            &self.ln_mlp_out_bias.shape(),
        );
        push_tensor_1d(
            tensors,
            &format!("{prefix}.mlp_out_norm.bias"),
            &self.ln_mlp_out_bias,
        )
        .await;
    }

    fn into_graph(self, graph: &Graph) -> Self {
        Self {
            ln_1_weight: regraph_tensor(graph, self.ln_1_weight),
            ln_1_bias: regraph_tensor(graph, self.ln_1_bias),
            mixer: self.mixer.into_graph(graph),
            ln_attn_out_weight: regraph_tensor(graph, self.ln_attn_out_weight),
            ln_attn_out_bias: regraph_tensor(graph, self.ln_attn_out_bias),
            ln_2_weight: regraph_tensor(graph, self.ln_2_weight),
            ln_2_bias: regraph_tensor(graph, self.ln_2_bias),
            mlp: self.mlp.into_graph(graph),
            ln_mlp_out_weight: regraph_tensor(graph, self.ln_mlp_out_weight),
            ln_mlp_out_bias: regraph_tensor(graph, self.ln_mlp_out_bias),
        }
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
            c_attn_k: random_matrix(graph, device, rng, shape.n_embd, shape.kv_dim(), init_scale),
            c_attn_v: random_matrix(graph, device, rng, shape.n_embd, shape.kv_dim(), init_scale),
            c_proj: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
        }
    }

    fn forward(
        &self,
        x: &Tensor<3>,
        causal_mask: &Tensor<3>,
        batch_size: usize,
        shape: ModelShape,
        rotary: Option<&RotaryEmbeddings>,
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
                .broadcast_as([batch_size, shape.n_embd, shape.kv_dim()]),
        );
        let v = x.mat_mul(
            &self
                .c_attn_v
                .broadcast_as([batch_size, shape.n_embd, shape.kv_dim()]),
        );
        let q_heads = (0..shape.n_head)
            .map(|head| {
                let start = head * head_dim;
                let end = start + head_dim;
                let head = q.slice([0..batch_size, 0..q.shape()[1], start..end]);
                rotary.map_or(head.clone(), |cache| cache.apply(&head))
            })
            .collect::<Vec<_>>();
        let k_heads = (0..shape.n_kv_head)
            .map(|head| {
                let start = head * head_dim;
                let end = start + head_dim;
                let head = k.slice([0..batch_size, 0..k.shape()[1], start..end]);
                rotary.map_or(head.clone(), |cache| cache.apply(&head))
            })
            .collect::<Vec<_>>();
        let v_heads = (0..shape.n_kv_head)
            .map(|head| {
                let start = head * head_dim;
                let end = start + head_dim;
                v.slice([0..batch_size, 0..v.shape()[1], start..end])
            })
            .collect::<Vec<_>>();
        let heads = (0..shape.n_head)
            .map(|head| {
                let kv_head = head / shape.num_kv_groups();
                let q_head = q_heads[head].clone();
                let k_head = k_heads[kv_head].clone();
                let v_head = v_heads[kv_head].clone();
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

    fn into_graph(self, graph: &Graph) -> Self {
        Self {
            c_attn_q: regraph_tensor(graph, self.c_attn_q),
            c_attn_k: regraph_tensor(graph, self.c_attn_k),
            c_attn_v: regraph_tensor(graph, self.c_attn_v),
            c_proj: regraph_tensor(graph, self.c_proj),
        }
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

    fn into_graph(self, graph: &Graph) -> Self {
        Self {
            kernels: self
                .kernels
                .into_iter()
                .map(|kernel| regraph_tensor(graph, kernel))
                .collect(),
            bias: regraph_tensor(graph, self.bias),
            out_proj: regraph_tensor(graph, self.out_proj),
        }
    }
}

impl SequenceMixer {
    fn forward(
        &self,
        x: &Tensor<3>,
        causal_mask: &Tensor<3>,
        batch_size: usize,
        shape: ModelShape,
        rotary: Option<&RotaryEmbeddings>,
    ) -> Tensor<3> {
        match self {
            SequenceMixer::Attention(attn) => {
                attn.forward(x, causal_mask, batch_size, shape, rotary)
            }
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

    fn into_graph(self, graph: &Graph) -> Self {
        match self {
            SequenceMixer::Attention(attn) => SequenceMixer::Attention(attn.into_graph(graph)),
            SequenceMixer::Conv(conv) => SequenceMixer::Conv(conv.into_graph(graph)),
        }
    }
}

impl RotaryEmbeddings {
    fn new(graph: &Graph, device: &Device, shape: ModelShape, rope_theta: f32) -> Self {
        let inverse_frequency = base_inverse_frequency(shape.head_dim(), rope_theta);
        let mut cos = vec![vec![0.0; inverse_frequency.len()]; shape.block_size];
        let mut sin = vec![vec![0.0; inverse_frequency.len()]; shape.block_size];
        for position in 0..shape.block_size {
            for (index, frequency) in inverse_frequency.iter().enumerate() {
                let angle = position as f32 * frequency;
                cos[position][index] = angle.cos();
                sin[position][index] = angle.sin();
            }
        }

        Self {
            cos: Tensor::constant_from_raw(graph, RawTensor::new(device, &cos)),
            sin: Tensor::constant_from_raw(graph, RawTensor::new(device, &sin)),
        }
    }

    fn apply(&self, x: &Tensor<3>) -> Tensor<3> {
        let [batch_size, sequence_length, head_dim] = x.shape();
        let half_dim = head_dim / 2;
        let cos = self
            .cos
            .slice([0..sequence_length, 0..half_dim])
            .unsqueeze(0)
            .broadcast_as([batch_size, sequence_length, half_dim]);
        let sin = self
            .sin
            .slice([0..sequence_length, 0..half_dim])
            .unsqueeze(0)
            .broadcast_as([batch_size, sequence_length, half_dim]);
        let first_half = x.slice([0..batch_size, 0..sequence_length, 0..half_dim]);
        let second_half = x.slice([0..batch_size, 0..sequence_length, half_dim..head_dim]);
        let rotated_first = first_half.mul(&cos).sub(&second_half.mul(&sin));
        let rotated_second = second_half.mul(&cos).add(&first_half.mul(&sin));
        Tensor::cat(vec![rotated_first, rotated_second], 2)
    }

    fn into_graph(self, graph: &Graph) -> Self {
        Self {
            cos: regraph_constant_tensor(graph, self.cos),
            sin: regraph_constant_tensor(graph, self.sin),
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

    fn into_graph(self, graph: &Graph) -> Self {
        Self {
            c_fc: regraph_tensor(graph, self.c_fc),
            c_fc_bias: regraph_tensor(graph, self.c_fc_bias),
            c_proj: regraph_tensor(graph, self.c_proj),
            c_proj_bias: regraph_tensor(graph, self.c_proj_bias),
        }
    }
}

impl AdamBlockState {
    fn new(device: &Device, block: &TransformerBlock) -> Self {
        Self {
            ln_1_weight: AdamMoments::zeros_like(device, &block.ln_1_weight),
            ln_1_bias: AdamMoments::zeros_like(device, &block.ln_1_bias),
            mixer: AdamMixerState::new(device, &block.mixer),
            ln_attn_out_weight: AdamMoments::zeros_like(device, &block.ln_attn_out_weight),
            ln_attn_out_bias: AdamMoments::zeros_like(device, &block.ln_attn_out_bias),
            ln_2_weight: AdamMoments::zeros_like(device, &block.ln_2_weight),
            ln_2_bias: AdamMoments::zeros_like(device, &block.ln_2_bias),
            mlp: AdamMlpState::new(device, &block.mlp),
            ln_mlp_out_weight: AdamMoments::zeros_like(device, &block.ln_mlp_out_weight),
            ln_mlp_out_bias: AdamMoments::zeros_like(device, &block.ln_mlp_out_bias),
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
            ln_attn_out_weight: adamw_update(
                &block.ln_attn_out_weight,
                &mut self.ln_attn_out_weight,
                gradients,
                step,
                settings,
            ),
            ln_attn_out_bias: adamw_update(
                &block.ln_attn_out_bias,
                &mut self.ln_attn_out_bias,
                gradients,
                step,
                settings,
            ),
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
            ln_mlp_out_weight: adamw_update(
                &block.ln_mlp_out_weight,
                &mut self.ln_mlp_out_weight,
                gradients,
                step,
                settings,
            ),
            ln_mlp_out_bias: adamw_update(
                &block.ln_mlp_out_bias,
                &mut self.ln_mlp_out_bias,
                gradients,
                step,
                settings,
            ),
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

fn regraph_tensor<const R: usize>(graph: &Graph, tensor: Tensor<R>) -> Tensor<R> {
    Tensor::from_raw(graph, tensor.into_raw())
}

fn regraph_constant_tensor<const R: usize>(graph: &Graph, tensor: Tensor<R>) -> Tensor<R> {
    Tensor::constant_from_raw(graph, tensor.into_raw())
}

fn adamw_update_optional<const R: usize>(
    parameter: Option<Tensor<R>>,
    moments: &mut Option<AdamMoments<R>>,
    gradients: &Gradients,
    step: usize,
    settings: AdamWSettings,
) -> Option<Tensor<R>> {
    match (parameter, moments.as_mut()) {
        (Some(parameter), Some(moments)) => {
            Some(adamw_update(&parameter, moments, gradients, step, settings))
        }
        (None, None) => None,
        _ => unreachable!("optimizer state does not match optional parameter"),
    }
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

    fn kv_dim(self) -> usize {
        self.n_kv_head * self.head_dim()
    }

    fn num_kv_groups(self) -> usize {
        self.n_head / self.n_kv_head.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{RuntimeConfig, SaveQuantization},
        data::StrokeTokenizer,
    };
    use rand::SeedableRng;

    #[tokio::test]
    async fn rope_mqa_model_uses_shared_kv_and_skips_position_embeddings() {
        let device = Device::cpu();
        let mut rng = StdRng::seed_from_u64(7);
        let config = RuntimeConfig {
            epochs: 1,
            warmup_steps: 1,
            learning_rate: 1e-3,
            min_learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.95,
            adam_eps: 1e-8,
            weight_decay: 0.01,
            log_every: 1,
            eval_batches: 1,
            sample_tokens: 8,
            sample_prefix_tokens: 4,
            sample_temperature: 0.7,
            sample_top_k: 4,
            block_size: 8,
            batch_size: 2,
            n_embd: 16,
            n_head: 4,
            n_kv_head: 1,
            n_ff: 32,
            n_layer: 1,
            conv_kernel_size: 3,
            attention_period: 1,
            use_rope: true,
            rope_theta: 10_000.0,
            use_canvas_state_embeddings: true,
            use_extra_norms: false,
            eps: 1e-5,
            init_scale: 0.05,
            seed: 7,
            save_every_steps: 0,
            save_final_model: false,
            save_quantization: SaveQuantization::F32,
            train_examples: 1,
            validation_examples: 1,
            test_examples: 1,
            dataset_path: None,
            include_synthetic_data: false,
            gguf_path: "test.gguf".into(),
            sample_output_path: "sample.svg".into(),
        };

        let tokenizer = StrokeTokenizer::new();
        let model = NanoChatModel::new(&device, &mut rng, &tokenizer, &config);
        assert_eq!(model.n_kv_head(), 1);
        assert!(model.use_rope());
        assert!(model.canvas_state_spec().is_some());

        let tensors = model.named_tensors().await;
        assert!(
            !tensors
                .iter()
                .any(|tensor| tensor.name == "position_embd.weight"),
            "RoPE models should not serialize learned position embeddings"
        );
        assert!(
            tensors
                .iter()
                .any(|tensor| tensor.name == "cursor_x_embd.weight"),
            "canvas-state models should serialize cursor x embeddings"
        );

        let attn_k = tensors
            .iter()
            .find(|tensor| tensor.name == "blk.0.attn_k.weight")
            .unwrap();
        let attn_v = tensors
            .iter()
            .find(|tensor| tensor.name == "blk.0.attn_v.weight")
            .unwrap();
        let cursor_x = tensors
            .iter()
            .find(|tensor| tensor.name == "cursor_x_embd.weight")
            .unwrap();
        assert_eq!(attn_k.shape, vec![16, 4]);
        assert_eq!(attn_v.shape, vec![16, 4]);
        assert_eq!(cursor_x.shape, vec![129, 16]);
    }
}
