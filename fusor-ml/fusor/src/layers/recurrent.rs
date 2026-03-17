use fusor_cpu::TensorBacking;

use crate::{Error, Result, Tensor};

#[derive(Clone)]
pub struct RecurrentWeights {
    input_proj: Tensor<2, f32>,
    state_proj: Tensor<2, f32>,
    gate_input_proj: Tensor<2, f32>,
    gate_state_proj: Tensor<2, f32>,
    out_proj: Tensor<2, f32>,
}

impl RecurrentWeights {
    pub fn new(
        input_proj: Tensor<2, f32>,
        state_proj: Tensor<2, f32>,
        gate_input_proj: Tensor<2, f32>,
        gate_state_proj: Tensor<2, f32>,
        out_proj: Tensor<2, f32>,
    ) -> Self {
        Self {
            input_proj,
            state_proj,
            gate_input_proj,
            gate_state_proj,
            out_proj,
        }
    }

    pub fn input_proj(&self) -> &Tensor<2, f32> {
        &self.input_proj
    }

    pub fn state_proj(&self) -> &Tensor<2, f32> {
        &self.state_proj
    }

    pub fn gate_input_proj(&self) -> &Tensor<2, f32> {
        &self.gate_input_proj
    }

    pub fn gate_state_proj(&self) -> &Tensor<2, f32> {
        &self.gate_state_proj
    }

    pub fn out_proj(&self) -> &Tensor<2, f32> {
        &self.out_proj
    }
}

pub fn recurrent_forward<B>(x: &Tensor<3, f32, B>, weights: &RecurrentWeights) -> Tensor<3, f32>
where
    B: TensorBacking<3, Elem = f32>,
{
    let [batch_size, seq_len, n_embd] = x.shape();
    debug_assert_eq!(weights.input_proj.shape(), [n_embd, n_embd]);
    debug_assert_eq!(weights.state_proj.shape(), [n_embd, n_embd]);
    debug_assert_eq!(weights.gate_input_proj.shape(), [n_embd, n_embd]);
    debug_assert_eq!(weights.gate_state_proj.shape(), [n_embd, n_embd]);
    debug_assert_eq!(weights.out_proj.shape(), [n_embd, n_embd]);

    let device = x.device();
    let ones: Tensor<2, f32> = Tensor::splat(&device, 1.0, [batch_size, n_embd]);
    let mut state: Tensor<2, f32> = Tensor::splat(&device, 0.0, [batch_size, n_embd]);
    let mut outputs = Vec::with_capacity(seq_len);

    for position in 0..seq_len {
        let x_t: Tensor<2, f32> = x
            .slice([0..batch_size, position..position + 1, 0..n_embd])
            .squeeze(1)
            .to_concrete();
        let candidate = (x_t.mat_mul(weights.input_proj()) + state.mat_mul(weights.state_proj()))
            .tanh()
            .to_concrete();
        let gate_pre =
            x_t.mat_mul(weights.gate_input_proj()) + state.mat_mul(weights.gate_state_proj());
        let gate = ((gate_pre.tanh() + &ones) * 0.5).to_concrete();
        let keep = (&ones - &gate).to_concrete();
        state = ((&gate * &candidate) + &(&keep * &state)).to_concrete();
        outputs.push(state.mat_mul(weights.out_proj()).to_concrete());
    }

    Tensor::stack(outputs, 1)
}

#[derive(Clone, Debug)]
pub struct HostRecurrentWeights {
    n_embd: usize,
    input_proj: Vec<f32>,
    state_proj: Vec<f32>,
    gate_input_proj: Vec<f32>,
    gate_state_proj: Vec<f32>,
    out_proj: Vec<f32>,
}

impl HostRecurrentWeights {
    pub fn from_nested(
        input_proj: Vec<Vec<f32>>,
        state_proj: Vec<Vec<f32>>,
        gate_input_proj: Vec<Vec<f32>>,
        gate_state_proj: Vec<Vec<f32>>,
        out_proj: Vec<Vec<f32>>,
    ) -> Result<Self> {
        let n_embd = square_matrix_size("input_proj", &input_proj)?;
        validate_square_matrix("state_proj", &state_proj, n_embd)?;
        validate_square_matrix("gate_input_proj", &gate_input_proj, n_embd)?;
        validate_square_matrix("gate_state_proj", &gate_state_proj, n_embd)?;
        validate_square_matrix("out_proj", &out_proj, n_embd)?;

        Ok(Self {
            n_embd,
            input_proj: flatten_matrix(&input_proj),
            state_proj: flatten_matrix(&state_proj),
            gate_input_proj: flatten_matrix(&gate_input_proj),
            gate_state_proj: flatten_matrix(&gate_state_proj),
            out_proj: flatten_matrix(&out_proj),
        })
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }
}

#[derive(Clone, Debug)]
struct HostRecurrentScan {
    batch_size: usize,
    seq_len: usize,
    n_embd: usize,
    prev_states: Vec<f32>,
    candidates: Vec<f32>,
    gates: Vec<f32>,
    states: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct HostRecurrentBackward {
    pub grad_input: Vec<f32>,
    pub grad_input_proj: Vec<f32>,
    pub grad_state_proj: Vec<f32>,
    pub grad_gate_input_proj: Vec<f32>,
    pub grad_gate_state_proj: Vec<f32>,
    pub grad_out_proj: Vec<f32>,
}

pub fn host_recurrent_forward(
    input: &[f32],
    batch_size: usize,
    seq_len: usize,
    weights: &HostRecurrentWeights,
) -> Result<Vec<f32>> {
    Ok(host_recurrent_scan(input, batch_size, seq_len, weights)?
        .states_to_outputs(&weights.out_proj))
}

pub fn host_recurrent_backward(
    input: &[f32],
    batch_size: usize,
    seq_len: usize,
    weights: &HostRecurrentWeights,
    grad_output: &[f32],
) -> Result<HostRecurrentBackward> {
    let scan = host_recurrent_scan(input, batch_size, seq_len, weights)?;
    let n_embd = weights.n_embd;
    let step_len = batch_size * n_embd;
    let total_len = batch_size * seq_len * n_embd;
    if grad_output.len() != total_len {
        return Err(Error::msg(format!(
            "expected grad_output length {}, got {}",
            total_len,
            grad_output.len()
        )));
    }

    let mut grad_state_next = vec![0.0; step_len];
    let mut grad_input_proj = vec![0.0; n_embd * n_embd];
    let mut grad_state_proj = vec![0.0; n_embd * n_embd];
    let mut grad_gate_input_proj = vec![0.0; n_embd * n_embd];
    let mut grad_gate_state_proj = vec![0.0; n_embd * n_embd];
    let mut grad_out_proj = vec![0.0; n_embd * n_embd];
    let mut grad_input = vec![0.0; total_len];

    for position in (0..seq_len).rev() {
        let grad_output_t = &grad_output[position * step_len..(position + 1) * step_len];
        let state = &scan.states[position * step_len..(position + 1) * step_len];
        let prev_state = &scan.prev_states[position * step_len..(position + 1) * step_len];
        let candidate = &scan.candidates[position * step_len..(position + 1) * step_len];
        let gate = &scan.gates[position * step_len..(position + 1) * step_len];
        let x_t = &input[position * step_len..(position + 1) * step_len];

        let grad_from_output =
            matmul_rhs_transposed(grad_output_t, &weights.out_proj, batch_size, n_embd);
        add_outer_product(&mut grad_out_proj, state, grad_output_t, batch_size, n_embd);

        let mut grad_state = vec![0.0; step_len];
        let mut grad_candidate_pre = vec![0.0; step_len];
        let mut grad_gate_pre = vec![0.0; step_len];
        let mut keep = vec![0.0; step_len];

        for index in 0..step_len {
            let grad_state_value = grad_from_output[index] + grad_state_next[index];
            let gate_value = gate[index];
            let candidate_value = candidate[index];
            let prev_state_value = prev_state[index];
            let keep_value = 1.0 - gate_value;
            let grad_candidate = grad_state_value * gate_value;
            let grad_gate = grad_state_value * (candidate_value - prev_state_value);
            let tanh_gate_pre = (gate_value * 2.0) - 1.0;

            grad_state[index] = grad_state_value;
            grad_candidate_pre[index] = grad_candidate * (1.0 - candidate_value * candidate_value);
            grad_gate_pre[index] = grad_gate * (1.0 - tanh_gate_pre * tanh_gate_pre) * 0.5;
            keep[index] = keep_value;
        }

        add_outer_product(
            &mut grad_input_proj,
            x_t,
            &grad_candidate_pre,
            batch_size,
            n_embd,
        );
        add_outer_product(
            &mut grad_state_proj,
            prev_state,
            &grad_candidate_pre,
            batch_size,
            n_embd,
        );
        add_outer_product(
            &mut grad_gate_input_proj,
            x_t,
            &grad_gate_pre,
            batch_size,
            n_embd,
        );
        add_outer_product(
            &mut grad_gate_state_proj,
            prev_state,
            &grad_gate_pre,
            batch_size,
            n_embd,
        );

        let grad_candidate_input =
            matmul_rhs_transposed(&grad_candidate_pre, &weights.input_proj, batch_size, n_embd);
        let grad_gate_input =
            matmul_rhs_transposed(&grad_gate_pre, &weights.gate_input_proj, batch_size, n_embd);
        for index in 0..step_len {
            grad_input[position * step_len + index] =
                grad_candidate_input[index] + grad_gate_input[index];
        }

        let grad_candidate_state =
            matmul_rhs_transposed(&grad_candidate_pre, &weights.state_proj, batch_size, n_embd);
        let grad_gate_state =
            matmul_rhs_transposed(&grad_gate_pre, &weights.gate_state_proj, batch_size, n_embd);
        for index in 0..step_len {
            grad_state_next[index] = (grad_state[index] * keep[index])
                + grad_candidate_state[index]
                + grad_gate_state[index];
        }
    }

    Ok(HostRecurrentBackward {
        grad_input,
        grad_input_proj,
        grad_state_proj,
        grad_gate_input_proj,
        grad_gate_state_proj,
        grad_out_proj,
    })
}

impl HostRecurrentScan {
    fn states_to_outputs(&self, out_proj: &[f32]) -> Vec<f32> {
        let mut outputs = vec![0.0; self.batch_size * self.seq_len * self.n_embd];
        let step_len = self.batch_size * self.n_embd;
        for position in 0..self.seq_len {
            let state = &self.states[position * step_len..(position + 1) * step_len];
            let output = matmul_row_major(state, out_proj, self.batch_size, self.n_embd);
            outputs[position * step_len..(position + 1) * step_len].copy_from_slice(&output);
        }
        outputs
    }
}

fn host_recurrent_scan(
    input: &[f32],
    batch_size: usize,
    seq_len: usize,
    weights: &HostRecurrentWeights,
) -> Result<HostRecurrentScan> {
    let n_embd = weights.n_embd;
    let total_len = batch_size * seq_len * n_embd;
    if input.len() != total_len {
        return Err(Error::msg(format!(
            "expected input length {}, got {}",
            total_len,
            input.len()
        )));
    }

    let step_len = batch_size * n_embd;
    let mut state = vec![0.0; step_len];
    let mut prev_states = vec![0.0; total_len];
    let mut candidates = vec![0.0; total_len];
    let mut gates = vec![0.0; total_len];
    let mut states = vec![0.0; total_len];

    for position in 0..seq_len {
        let step_offset = position * step_len;
        let x_t = &input[step_offset..step_offset + step_len];
        prev_states[step_offset..step_offset + step_len].copy_from_slice(&state);

        let input_candidate = matmul_row_major(x_t, &weights.input_proj, batch_size, n_embd);
        let state_candidate = matmul_row_major(&state, &weights.state_proj, batch_size, n_embd);
        let input_gate = matmul_row_major(x_t, &weights.gate_input_proj, batch_size, n_embd);
        let state_gate = matmul_row_major(&state, &weights.gate_state_proj, batch_size, n_embd);

        for index in 0..step_len {
            let candidate = (input_candidate[index] + state_candidate[index]).tanh();
            let gate = ((input_gate[index] + state_gate[index]).tanh() + 1.0) * 0.5;
            let next_state = (gate * candidate) + ((1.0 - gate) * state[index]);
            candidates[step_offset + index] = candidate;
            gates[step_offset + index] = gate;
            states[step_offset + index] = next_state;
            state[index] = next_state;
        }
    }

    Ok(HostRecurrentScan {
        batch_size,
        seq_len,
        n_embd,
        prev_states,
        candidates,
        gates,
        states,
    })
}

fn square_matrix_size(name: &str, matrix: &[Vec<f32>]) -> Result<usize> {
    let size = matrix.len();
    validate_square_matrix(name, matrix, size)?;
    Ok(size)
}

fn validate_square_matrix(name: &str, matrix: &[Vec<f32>], expected: usize) -> Result<()> {
    if matrix.len() != expected {
        return Err(Error::msg(format!(
            "{name} expected {expected} rows, got {}",
            matrix.len()
        )));
    }
    for (row_index, row) in matrix.iter().enumerate() {
        if row.len() != expected {
            return Err(Error::msg(format!(
                "{name} row {row_index} expected {expected} columns, got {}",
                row.len()
            )));
        }
    }
    Ok(())
}

fn flatten_matrix(matrix: &[Vec<f32>]) -> Vec<f32> {
    matrix.iter().flat_map(|row| row.iter().copied()).collect()
}

fn matmul_row_major(lhs: &[f32], rhs: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0; rows * cols];
    for row in 0..rows {
        for out_col in 0..cols {
            let mut sum = 0.0;
            for inner in 0..cols {
                sum += lhs[(row * cols) + inner] * rhs[(inner * cols) + out_col];
            }
            output[(row * cols) + out_col] = sum;
        }
    }
    output
}

fn matmul_rhs_transposed(lhs: &[f32], rhs: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0; rows * cols];
    for row in 0..rows {
        for out_col in 0..cols {
            let mut sum = 0.0;
            for inner in 0..cols {
                sum += lhs[(row * cols) + inner] * rhs[(out_col * cols) + inner];
            }
            output[(row * cols) + out_col] = sum;
        }
    }
    output
}

fn add_outer_product(accumulator: &mut [f32], lhs: &[f32], rhs: &[f32], rows: usize, cols: usize) {
    for row in 0..rows {
        let lhs_row = &lhs[row * cols..(row + 1) * cols];
        let rhs_row = &rhs[row * cols..(row + 1) * cols];
        for lhs_col in 0..cols {
            for rhs_col in 0..cols {
                accumulator[(lhs_col * cols) + rhs_col] += lhs_row[lhs_col] * rhs_row[rhs_col];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_weights() -> HostRecurrentWeights {
        HostRecurrentWeights::from_nested(
            vec![vec![0.2, -0.1], vec![0.05, 0.15]],
            vec![vec![0.1, 0.03], vec![-0.04, 0.07]],
            vec![vec![-0.15, 0.08], vec![0.02, -0.05]],
            vec![vec![0.06, -0.02], vec![0.04, 0.09]],
            vec![vec![0.3, -0.12], vec![0.11, 0.21]],
        )
        .unwrap()
    }

    fn loss(
        input: &[f32],
        batch_size: usize,
        seq_len: usize,
        weights: &HostRecurrentWeights,
        grad_output: &[f32],
    ) -> f32 {
        host_recurrent_forward(input, batch_size, seq_len, weights)
            .unwrap()
            .into_iter()
            .zip(grad_output.iter().copied())
            .map(|(output, grad)| output * grad)
            .sum()
    }

    #[tokio::test]
    async fn test_recurrent_forward_matches_host_reference_on_cpu() {
        let weights = RecurrentWeights::new(
            Tensor::Cpu(fusor_cpu::Tensor::from_slice(
                [2, 2],
                &[0.2, -0.1, 0.05, 0.15],
            )),
            Tensor::Cpu(fusor_cpu::Tensor::from_slice(
                [2, 2],
                &[0.1, 0.03, -0.04, 0.07],
            )),
            Tensor::Cpu(fusor_cpu::Tensor::from_slice(
                [2, 2],
                &[-0.15, 0.08, 0.02, -0.05],
            )),
            Tensor::Cpu(fusor_cpu::Tensor::from_slice(
                [2, 2],
                &[0.06, -0.02, 0.04, 0.09],
            )),
            Tensor::Cpu(fusor_cpu::Tensor::from_slice(
                [2, 2],
                &[0.3, -0.12, 0.11, 0.21],
            )),
        );
        let input: Tensor<3, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [1, 2, 2],
            &[0.25, -0.4, 0.1, 0.3],
        ));

        let output = recurrent_forward(&input, &weights)
            .as_slice()
            .await
            .unwrap();
        let expected =
            host_recurrent_forward(&[0.25, -0.4, 0.1, 0.3], 1, 2, &test_weights()).unwrap();

        for (actual, expected) in output.as_slice().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_host_recurrent_backward_matches_finite_difference() {
        let weights = test_weights();
        let input = vec![0.25, -0.4, 0.1, 0.3];
        let grad_output = vec![0.7, -0.25, 0.1, 0.9];
        let grads = host_recurrent_backward(&input, 1, 2, &weights, &grad_output).unwrap();
        let eps = 1e-3;

        for index in 0..input.len() {
            let mut plus = input.clone();
            plus[index] += eps;
            let mut minus = input.clone();
            minus[index] -= eps;
            let numerical = (loss(&plus, 1, 2, &weights, &grad_output)
                - loss(&minus, 1, 2, &weights, &grad_output))
                / (2.0 * eps);
            assert!((grads.grad_input[index] - numerical).abs() < 2e-3);
        }

        for index in 0..weights.input_proj.len() {
            let mut plus_weights = weights.clone();
            let mut minus_weights = weights.clone();
            plus_weights.input_proj[index] += eps;
            minus_weights.input_proj[index] -= eps;
            let numerical = (loss(&input, 1, 2, &plus_weights, &grad_output)
                - loss(&input, 1, 2, &minus_weights, &grad_output))
                / (2.0 * eps);
            assert!((grads.grad_input_proj[index] - numerical).abs() < 2e-3);
        }

        for index in 0..weights.state_proj.len() {
            let mut plus_weights = weights.clone();
            let mut minus_weights = weights.clone();
            plus_weights.state_proj[index] += eps;
            minus_weights.state_proj[index] -= eps;
            let numerical = (loss(&input, 1, 2, &plus_weights, &grad_output)
                - loss(&input, 1, 2, &minus_weights, &grad_output))
                / (2.0 * eps);
            assert!((grads.grad_state_proj[index] - numerical).abs() < 2e-3);
        }

        for index in 0..weights.gate_input_proj.len() {
            let mut plus_weights = weights.clone();
            let mut minus_weights = weights.clone();
            plus_weights.gate_input_proj[index] += eps;
            minus_weights.gate_input_proj[index] -= eps;
            let numerical = (loss(&input, 1, 2, &plus_weights, &grad_output)
                - loss(&input, 1, 2, &minus_weights, &grad_output))
                / (2.0 * eps);
            assert!((grads.grad_gate_input_proj[index] - numerical).abs() < 2e-3);
        }

        for index in 0..weights.gate_state_proj.len() {
            let mut plus_weights = weights.clone();
            let mut minus_weights = weights.clone();
            plus_weights.gate_state_proj[index] += eps;
            minus_weights.gate_state_proj[index] -= eps;
            let numerical = (loss(&input, 1, 2, &plus_weights, &grad_output)
                - loss(&input, 1, 2, &minus_weights, &grad_output))
                / (2.0 * eps);
            assert!((grads.grad_gate_state_proj[index] - numerical).abs() < 2e-3);
        }

        for index in 0..weights.out_proj.len() {
            let mut plus_weights = weights.clone();
            let mut minus_weights = weights.clone();
            plus_weights.out_proj[index] += eps;
            minus_weights.out_proj[index] -= eps;
            let numerical = (loss(&input, 1, 2, &plus_weights, &grad_output)
                - loss(&input, 1, 2, &minus_weights, &grad_output))
                / (2.0 * eps);
            assert!((grads.grad_out_proj[index] - numerical).abs() < 2e-3);
        }
    }
}
