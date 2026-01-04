use crate::{Device, Result, Tensor, VarBuilder};

/// Activation function for the GRU hidden state
#[derive(Clone, Copy, Debug, Default)]
pub enum GruActivation {
    #[default]
    Tanh,
    Sigmoid,
    Relu,
}

/// A GRU (Gated Recurrent Unit) layer.
///
/// This implements the standard GRU equations:
/// - z = sigmoid(W_z * x + U_z * h + b_z)  // update gate
/// - r = sigmoid(W_r * x + U_r * h + b_r)  // reset gate
/// - h_candidate = activation(W_h * x + U_h * (r * h) + b_h)
/// - h_new = z * h + (1 - z) * h_candidate
///
/// The weights are stored with gates interleaved (z, r, h) along the first dimension.
pub struct Gru {
    /// Input weights [3 * hidden_size, input_size]
    input_weights: Tensor<2, f32>,
    /// Recurrent weights [3 * hidden_size, hidden_size]
    recurrent_weights: Tensor<2, f32>,
    /// Bias [3 * hidden_size]
    bias: Tensor<1, f32>,
    /// Hidden size
    hidden_size: usize,
    /// Activation function for the hidden state candidate
    activation: GruActivation,
}

impl Gru {
    /// Create a new GRU layer from tensors
    pub fn new(
        input_weights: Tensor<2, f32>,
        recurrent_weights: Tensor<2, f32>,
        bias: Tensor<1, f32>,
        hidden_size: usize,
        activation: GruActivation,
    ) -> Self {
        // Validate shapes
        let iw_shape = input_weights.shape();
        let rw_shape = recurrent_weights.shape();
        let b_shape = bias.shape();

        assert_eq!(
            iw_shape[0],
            3 * hidden_size,
            "Input weights first dim should be 3 * hidden_size"
        );
        assert_eq!(
            rw_shape[0],
            3 * hidden_size,
            "Recurrent weights first dim should be 3 * hidden_size"
        );
        assert_eq!(
            rw_shape[1], hidden_size,
            "Recurrent weights second dim should be hidden_size"
        );
        assert_eq!(
            b_shape[0],
            3 * hidden_size,
            "Bias should have 3 * hidden_size elements"
        );

        Self {
            input_weights,
            recurrent_weights,
            bias,
            hidden_size,
            activation,
        }
    }

    /// Load a GRU layer from a VarBuilder
    ///
    /// Expects the following tensors:
    /// - "input_weights": [3 * hidden_size, input_size]
    /// - "recurrent_weights": [3 * hidden_size, hidden_size]
    /// - "bias": [3 * hidden_size]
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        hidden_size: usize,
        activation: GruActivation,
    ) -> Result<Self> {
        let input_weights = vb.get("input_weights", device)?.dequantize();
        let recurrent_weights = vb.get("recurrent_weights", device)?.dequantize();
        let bias = vb.get("bias", device)?.dequantize();

        Ok(Self::new(
            input_weights,
            recurrent_weights,
            bias,
            hidden_size,
            activation,
        ))
    }

    /// Get the hidden size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Forward pass through the GRU
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch, input_size]
    /// * `state` - Previous hidden state [batch, hidden_size]
    ///
    /// # Returns
    /// New hidden state [batch, hidden_size]
    pub fn forward(&self, input: &Tensor<2, f32>, state: &Tensor<2, f32>) -> Tensor<2, f32> {
        let h = self.hidden_size;

        // Slice weights for each gate
        // input_weights: [3*h, input_size], we need [h, input_size] for each gate
        let w_z = self.input_weights.narrow(0, 0, h);
        let w_r = self.input_weights.narrow(0, h, h);
        let w_h = self.input_weights.narrow(0, 2 * h, h);

        // recurrent_weights: [3*h, h], we need [h, h] for each gate
        let u_z = self.recurrent_weights.narrow(0, 0, h);
        let u_r = self.recurrent_weights.narrow(0, h, h);
        let u_h = self.recurrent_weights.narrow(0, 2 * h, h);

        // bias: [3*h], we need [h] for each gate
        let b_z = self.bias.narrow(0, 0, h);
        let b_r = self.bias.narrow(0, h, h);
        let b_h = self.bias.narrow(0, 2 * h, h);

        // Compute update gate: z = sigmoid(W_z * x + U_z * h + b_z)
        // input: [batch, input_size], w_z.f32: [input_size, h] -> result: [batch, h]
        let z = input
            .mat_mul(&w_z.transpose(0, 1))
            .add_(&state.mat_mul(&u_z.transpose(0, 1)))
            .add_(&b_z)
            .sigmoid();

        // Compute reset gate: r = sigmoid(W_r * x + U_r * h + b_r)
        let r = input
            .mat_mul(&w_r.transpose(0, 1))
            .add_(&state.mat_mul(&u_r.transpose(0, 1)))
            .add_(&b_r)
            .sigmoid();

        // Compute hidden candidate: h_candidate = activation(W_h * x + U_h * (r * h) + b_h)
        let r_state: Tensor<2, f32> = &r * state;
        let h_candidate_pre = input
            .mat_mul(&w_h.transpose(0, 1))
            .add_(&r_state.mat_mul(&u_h.transpose(0, 1)))
            .add_(&b_h);

        let h_candidate = match self.activation {
            GruActivation::Tanh => h_candidate_pre.tanh(),
            GruActivation::Sigmoid => h_candidate_pre.sigmoid(),
            GruActivation::Relu => h_candidate_pre.relu(),
        };

        // Compute new state: h_new = z * h + (1 - z) * h_candidate
        // = z * h + h_candidate - z * h_candidate
        // = h_candidate + z * (h - h_candidate)
        let state_diff: Tensor<2, f32> = state - &h_candidate;
        let z_diff: Tensor<2, f32> = &z * &state_diff;
        &h_candidate + &z_diff
    }
}
