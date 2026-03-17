use fusor::{
    Device, Tensor, VarBuilder, cache::AttentionMask,
    layers::{Embedding, RecurrentWeights, recurrent_forward},
};

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
pub struct InteractiveNanoChatModel {
    shape: ModelShape,
    attention_period: usize,
    vocab_size: usize,
    wte: Tensor<2, f32>,
    wpe: Tensor<2, f32>,
    blocks: Vec<TransformerBlock>,
    ln_f_weight: Tensor<1, f32>,
    ln_f_bias: Tensor<1, f32>,
    lm_head: Tensor<2, f32>,
}

#[derive(Clone)]
struct TransformerBlock {
    ln_1_weight: Tensor<1, f32>,
    ln_1_bias: Tensor<1, f32>,
    mixer: SequenceMixer,
    ln_2_weight: Tensor<1, f32>,
    ln_2_bias: Tensor<1, f32>,
    mlp: Mlp,
}

#[derive(Clone)]
enum SequenceMixer {
    Attention(CausalSelfAttention),
    Conv(ConvMixer),
    Recurrent(RecurrentMixer),
}

#[derive(Clone)]
struct CausalSelfAttention {
    c_attn_q: Tensor<2, f32>,
    c_attn_k: Tensor<2, f32>,
    c_attn_v: Tensor<2, f32>,
    c_proj: Tensor<2, f32>,
}

#[derive(Clone)]
struct RecurrentMixer {
    input_proj: Tensor<2, f32>,
    state_proj: Tensor<2, f32>,
    gate_input_proj: Tensor<2, f32>,
    gate_state_proj: Tensor<2, f32>,
    out_proj: Tensor<2, f32>,
}

#[derive(Clone)]
struct ConvMixer {
    kernels: Vec<Tensor<2, f32>>,
    bias: Tensor<1, f32>,
    out_proj: Tensor<2, f32>,
}

#[derive(Clone)]
struct Mlp {
    c_fc: Tensor<2, f32>,
    c_fc_bias: Tensor<1, f32>,
    c_proj: Tensor<2, f32>,
    c_proj_bias: Tensor<1, f32>,
}

impl InteractiveNanoChatModel {
    pub fn load(device: &Device, vb: &mut VarBuilder) -> fusor::Result<Self> {
        let block_size = metadata_u32(vb, "nanochat.block_size")? as usize;
        let n_embd = metadata_u32(vb, "nanochat.embedding_length")? as usize;
        let n_head = vb
            .get_metadata("nanochat.head_count")
            .and_then(|value| value.to_u32().ok())
            .map(|value| value as usize)
            .unwrap_or(1)
            .max(1);
        let n_ff = metadata_u32(vb, "nanochat.feed_forward_length")? as usize;
        let conv_kernel_size = vb
            .get_metadata("nanochat.conv_kernel_size")
            .and_then(|value| value.to_u32().ok())
            .map(|value| value as usize)
            .unwrap_or(3)
            .max(1);
        let n_layer = metadata_u32(vb, "nanochat.block_count")? as usize;
        let attention_period = vb
            .get_metadata("nanochat.attention_period")
            .and_then(|value| value.to_u32().ok())
            .map(|value| value as usize)
            .unwrap_or(4)
            .max(1);
        let vocab_size = metadata_u32(vb, "nanochat.vocab_size")? as usize;
        let eps = metadata_f32(vb, "nanochat.eps")?;
        let shape = ModelShape {
            block_size,
            n_embd,
            n_head,
            n_ff,
            conv_kernel_size,
            eps,
        };
        assert_eq!(
            shape.n_embd % shape.n_head,
            0,
            "checkpoint embedding length {} must be divisible by head count {}",
            shape.n_embd,
            shape.n_head
        );

        let wte: Tensor<2, f32> = vb.get("token_embd.weight", device)?.dequantize();
        let wpe: Tensor<2, f32> = vb.get("position_embd.weight", device)?.dequantize();
        let blocks = (0..n_layer)
            .map(|index| {
                TransformerBlock::load(
                    device,
                    &mut vb.pp(format!("blk.{index}")),
                    is_attention_layer(index, attention_period),
                    shape,
                )
            })
            .collect::<fusor::Result<Vec<_>>>()?;
        let ln_f_weight: Tensor<1, f32> = vb.get("output_norm.weight", device)?.dequantize();
        let ln_f_bias: Tensor<1, f32> = vb.get("output_norm.bias", device)?.dequantize();
        let lm_head: Tensor<2, f32> = vb.get("output.weight", device)?.dequantize();

        Ok(Self {
            shape,
            attention_period,
            vocab_size,
            wte,
            wpe,
            blocks,
            ln_f_weight,
            ln_f_bias,
            lm_head,
        })
    }

    pub fn block_size(&self) -> usize {
        self.shape.block_size
    }

    pub fn attention_period(&self) -> usize {
        self.attention_period
    }

    #[allow(dead_code)]
    pub fn forward(
        &self,
        token_inputs: &Tensor<2, u32>,
        position_inputs: &Tensor<2, u32>,
        causal_mask: &AttentionMask<f32>,
    ) -> Tensor<3, f32> {
        let batch_size = token_inputs.shape()[0];
        let token_embeddings: Tensor<3, f32> =
            Embedding::new_from_tensor(self.wte.clone()).forward(token_inputs);
        let position_embeddings: Tensor<3, f32> =
            Embedding::new_from_tensor(self.wpe.clone()).forward(position_inputs);
        let mut x: Tensor<3, f32> = (token_embeddings + position_embeddings).to_concrete();

        for block in &self.blocks {
            x = block.forward(x, causal_mask, batch_size, self.shape);
        }

        let ln_f_weight = self.ln_f_weight.broadcast_as(x.shape());
        let ln_f_bias = self.ln_f_bias.broadcast_as(x.shape());
        let x = x
            .layer_norm(&ln_f_weight, Some(&ln_f_bias), self.shape.eps, true)
            .to_concrete();
        x.mat_mul(&self.lm_head.broadcast_as([batch_size, self.shape.n_embd, self.vocab_size]))
    }
}

impl TransformerBlock {
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        is_attention_layer: bool,
        shape: ModelShape,
    ) -> fusor::Result<Self> {
        let ln_1_weight: Tensor<1, f32> = vb.get("ln_1.weight", device)?.dequantize();
        let ln_1_bias: Tensor<1, f32> = vb.get("ln_1.bias", device)?.dequantize();
        let mixer = if is_attention_layer {
            SequenceMixer::Attention(CausalSelfAttention::load(device, vb)?)
        } else if vb.contains_key("conv_kernel.0.weight") {
            SequenceMixer::Conv(ConvMixer::load(device, vb, shape)?)
        } else {
            SequenceMixer::Recurrent(RecurrentMixer::load(device, vb)?)
        };
        let ln_2_weight: Tensor<1, f32> = vb.get("ln_2.weight", device)?.dequantize();
        let ln_2_bias: Tensor<1, f32> = vb.get("ln_2.bias", device)?.dequantize();
        let mlp = Mlp::load(device, vb)?;
        Ok(Self {
            ln_1_weight,
            ln_1_bias,
            mixer,
            ln_2_weight,
            ln_2_bias,
            mlp,
        })
    }

    fn forward(
        &self,
        x: Tensor<3, f32>,
        causal_mask: &AttentionMask<f32>,
        batch_size: usize,
        shape: ModelShape,
    ) -> Tensor<3, f32> {
        let ln_1_weight = self.ln_1_weight.broadcast_as(x.shape());
        let ln_1_bias = self.ln_1_bias.broadcast_as(x.shape());
        let attn_input = x
            .layer_norm(&ln_1_weight, Some(&ln_1_bias), shape.eps, true)
            .to_concrete();
        let attn_output = self.mixer.forward(&attn_input, causal_mask, batch_size, shape);
        let x: Tensor<3, f32> = (x + attn_output).to_concrete();

        let ln_2_weight = self.ln_2_weight.broadcast_as(x.shape());
        let ln_2_bias = self.ln_2_bias.broadcast_as(x.shape());
        let mlp_input = x
            .layer_norm(&ln_2_weight, Some(&ln_2_bias), shape.eps, true)
            .to_concrete();
        (x + self.mlp.forward(&mlp_input, batch_size, shape)).to_concrete()
    }
}

impl CausalSelfAttention {
    fn load(device: &Device, vb: &mut VarBuilder) -> fusor::Result<Self> {
        Ok(Self {
            c_attn_q: get_tensor2(vb, device, &["attn_q.weight", "attn.attn_q.weight"])?,
            c_attn_k: get_tensor2(vb, device, &["attn_k.weight", "attn.attn_k.weight"])?,
            c_attn_v: get_tensor2(vb, device, &["attn_v.weight", "attn.attn_v.weight"])?,
            c_proj: get_tensor2(vb, device, &["attn_proj.weight", "attn.attn_proj.weight"])?,
        })
    }

    fn forward(
        &self,
        x: &Tensor<3, f32>,
        causal_mask: &AttentionMask<f32>,
        batch_size: usize,
        shape: ModelShape,
    ) -> Tensor<3, f32> {
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
                let masked: Tensor<3, f32> = causal_mask.apply(&scores).to_concrete();
                let weights_exp = masked.exp();
                let attention = weights_exp.div_(&weights_exp.sum_keepdim::<2>(2));
                attention.mat_mul(&v_head).to_concrete()
            })
            .collect::<Vec<_>>();

        Tensor::cat(heads, 2)
            .mat_mul(&self.c_proj.broadcast_as([batch_size, shape.n_embd, shape.n_embd]))
            .to_concrete()
    }
}

impl SequenceMixer {
    fn forward(
        &self,
        x: &Tensor<3, f32>,
        causal_mask: &AttentionMask<f32>,
        batch_size: usize,
        shape: ModelShape,
    ) -> Tensor<3, f32> {
        match self {
            SequenceMixer::Attention(attn) => attn.forward(x, causal_mask, batch_size, shape),
            SequenceMixer::Conv(conv) => conv.forward(x, batch_size, shape),
            SequenceMixer::Recurrent(recurrent) => recurrent.forward(x, batch_size, shape),
        }
    }
}

impl RecurrentMixer {
    fn load(device: &Device, vb: &mut VarBuilder) -> fusor::Result<Self> {
        Ok(Self {
            input_proj: get_tensor2(vb, device, &["recurrent_in.weight"])?,
            state_proj: get_tensor2(vb, device, &["recurrent_state.weight"])?,
            gate_input_proj: get_tensor2(vb, device, &["recurrent_gate_in.weight"])?,
            gate_state_proj: get_tensor2(vb, device, &["recurrent_gate_state.weight"])?,
            out_proj: get_tensor2(vb, device, &["recurrent_out.weight"])?,
        })
    }

    fn forward(&self, x: &Tensor<3, f32>, batch_size: usize, shape: ModelShape) -> Tensor<3, f32> {
        debug_assert_eq!(batch_size, x.shape()[0]);
        debug_assert_eq!(shape.n_embd, x.shape()[2]);
        recurrent_forward(
            x,
            &RecurrentWeights::new(
                self.input_proj.clone(),
                self.state_proj.clone(),
                self.gate_input_proj.clone(),
                self.gate_state_proj.clone(),
                self.out_proj.clone(),
            ),
        )
    }
}

impl ConvMixer {
    fn load(device: &Device, vb: &mut VarBuilder, shape: ModelShape) -> fusor::Result<Self> {
        let mut kernels = Vec::with_capacity(shape.conv_kernel_size);
        for index in 0..shape.conv_kernel_size {
            let key = format!("conv_kernel.{index}.weight");
            let tensor = if vb.contains_key(&key) {
                vb.get(&key, device)?.dequantize()
            } else {
                return Err(fusor::Error::VarBuilder(format!(
                    "missing conv kernel in GGUF metadata: {key}"
                )));
            };
            kernels.push(tensor);
        }
        Ok(Self {
            kernels,
            bias: get_tensor1(vb, device, &["conv_bias"])?,
            out_proj: get_tensor2(vb, device, &["conv_proj.weight"])?,
        })
    }

    fn forward(&self, x: &Tensor<3, f32>, batch_size: usize, shape: ModelShape) -> Tensor<3, f32> {
        let seq_len = x.shape()[1];
        let mut mixed: Tensor<3, f32> = Tensor::zeros(&x.device(), [batch_size, seq_len, shape.n_embd]);

        for (offset, kernel) in self.kernels.iter().enumerate() {
            let shifted = causal_shift(x, offset);
            let projected =
                shifted.mat_mul(&kernel.broadcast_as([batch_size, shape.n_embd, shape.n_embd]));
            mixed = (mixed + projected).to_concrete();
        }

        mixed
            .add_(&self.bias.broadcast_as([batch_size, seq_len, shape.n_embd]))
            .relu()
            .mat_mul(&self.out_proj.broadcast_as([batch_size, shape.n_embd, shape.n_embd]))
            .to_concrete()
    }
}

impl Mlp {
    fn load(device: &Device, vb: &mut VarBuilder) -> fusor::Result<Self> {
        Ok(Self {
            c_fc: get_tensor2(vb, device, &["mlp_fc.weight", "mlp.mlp_fc.weight"])?,
            c_fc_bias: get_tensor1(vb, device, &["mlp_fc.bias", "mlp.mlp_fc.bias"])?,
            c_proj: get_tensor2(vb, device, &["mlp_proj.weight", "mlp.mlp_proj.weight"])?,
            c_proj_bias: get_tensor1(vb, device, &["mlp_proj.bias", "mlp.mlp_proj.bias"])?,
        })
    }

    fn forward(&self, x: &Tensor<3, f32>, batch_size: usize, shape: ModelShape) -> Tensor<3, f32> {
        let hidden = x
            .mat_mul(&self.c_fc.broadcast_as([batch_size, shape.n_embd, shape.n_ff]))
            .add_(&self.c_fc_bias)
            .relu()
            .to_concrete();

        hidden
            .mat_mul(&self.c_proj.broadcast_as([batch_size, shape.n_ff, shape.n_embd]))
            .add_(&self.c_proj_bias)
            .to_concrete()
    }
}

fn metadata_u32(vb: &VarBuilder, key: &str) -> fusor::Result<u32> {
    vb.get_metadata(key)
        .ok_or_else(|| fusor::Error::msg(format!("Key '{key}' not found in GGUF metadata")))?
        .to_u32()
        .map_err(|error| fusor::Error::msg(error.to_string()))
}

fn metadata_f32(vb: &VarBuilder, key: &str) -> fusor::Result<f32> {
    vb.get_metadata(key)
        .ok_or_else(|| fusor::Error::msg(format!("Key '{key}' not found in GGUF metadata")))?
        .to_f32()
        .map_err(|error| fusor::Error::msg(error.to_string()))
}

fn get_tensor1(vb: &mut VarBuilder, device: &Device, keys: &[&str]) -> fusor::Result<Tensor<1, f32>> {
    for key in keys {
        if vb.contains_key(key) {
            return Ok(vb.get(key, device)?.dequantize());
        }
    }
    Err(fusor::Error::VarBuilder(format!(
        "none of the candidate keys were found in GGUF metadata: {}",
        keys.join(", ")
    )))
}

fn get_tensor2(vb: &mut VarBuilder, device: &Device, keys: &[&str]) -> fusor::Result<Tensor<2, f32>> {
    for key in keys {
        if vb.contains_key(key) {
            return Ok(vb.get(key, device)?.dequantize());
        }
    }
    Err(fusor::Error::VarBuilder(format!(
        "none of the candidate keys were found in GGUF metadata: {}",
        keys.join(", ")
    )))
}

fn is_attention_layer(index: usize, attention_period: usize) -> bool {
    (index + 1) % attention_period.max(1) == 0
}

fn causal_shift(x: &Tensor<3, f32>, offset: usize) -> Tensor<3, f32> {
    if offset == 0 {
        return x.clone();
    }

    let [batch_size, seq_len, n_embd] = x.shape();
    if offset >= seq_len {
        return Tensor::zeros(&x.device(), [batch_size, seq_len, n_embd]);
    }

    let prefix: Tensor<3, f32> = Tensor::zeros(&x.device(), [batch_size, offset, n_embd]);
    let shifted = x
        .slice([0..batch_size, 0..seq_len - offset, 0..n_embd])
        .to_concrete();
    Tensor::cat(vec![prefix, shifted], 1).to_concrete()
}

impl ModelShape {
    fn head_dim(self) -> usize {
        self.n_embd / self.n_head.max(1)
    }
}
