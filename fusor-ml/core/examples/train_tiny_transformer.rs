use fusor_core::{Device, Gradients, Tensor, cache::AttentionMask};
use rand::{Rng, SeedableRng, rngs::StdRng};

const VOCAB_SIZE: usize = 6;
const SEQ_LEN: usize = 4;
const BATCH_SIZE: usize = 6;
const MODEL_DIM: usize = 8;
const FF_DIM: usize = 16;
const EPOCHS: usize = 120;
const LEARNING_RATE: f32 = 0.08;
const EPS: f32 = 1e-5;

#[derive(Clone)]
struct TinyTransformer {
    token_projection: Tensor<2, f32>,
    position_projection: Tensor<2, f32>,
    ln1_weight: Tensor<1, f32>,
    ln1_bias: Tensor<1, f32>,
    w_q: Tensor<2, f32>,
    w_k: Tensor<2, f32>,
    w_v: Tensor<2, f32>,
    w_o: Tensor<2, f32>,
    ln2_weight: Tensor<1, f32>,
    ln2_bias: Tensor<1, f32>,
    w1: Tensor<2, f32>,
    b1: Tensor<1, f32>,
    w2: Tensor<2, f32>,
    b2: Tensor<1, f32>,
    ln_out_weight: Tensor<1, f32>,
    ln_out_bias: Tensor<1, f32>,
    lm_head: Tensor<2, f32>,
}

impl TinyTransformer {
    fn new(device: &Device) -> Self {
        let mut rng = StdRng::seed_from_u64(7);
        Self {
            token_projection: random_matrix::<VOCAB_SIZE, MODEL_DIM>(device, &mut rng, 0.12),
            position_projection: random_matrix::<SEQ_LEN, MODEL_DIM>(device, &mut rng, 0.12),
            ln1_weight: ones::<MODEL_DIM>(device),
            ln1_bias: zeros::<MODEL_DIM>(device),
            w_q: random_matrix::<MODEL_DIM, MODEL_DIM>(device, &mut rng, 0.10),
            w_k: random_matrix::<MODEL_DIM, MODEL_DIM>(device, &mut rng, 0.10),
            w_v: random_matrix::<MODEL_DIM, MODEL_DIM>(device, &mut rng, 0.10),
            w_o: random_matrix::<MODEL_DIM, MODEL_DIM>(device, &mut rng, 0.10),
            ln2_weight: ones::<MODEL_DIM>(device),
            ln2_bias: zeros::<MODEL_DIM>(device),
            w1: random_matrix::<MODEL_DIM, FF_DIM>(device, &mut rng, 0.10),
            b1: zeros::<FF_DIM>(device),
            w2: random_matrix::<FF_DIM, MODEL_DIM>(device, &mut rng, 0.10),
            b2: zeros::<MODEL_DIM>(device),
            ln_out_weight: ones::<MODEL_DIM>(device),
            ln_out_bias: zeros::<MODEL_DIM>(device),
            lm_head: random_matrix::<MODEL_DIM, VOCAB_SIZE>(device, &mut rng, 0.10),
        }
    }

    fn forward(
        &self,
        token_inputs: &Tensor<3, f32>,
        position_inputs: &Tensor<2, f32>,
        causal_mask: &AttentionMask<f32>,
    ) -> Tensor<3, f32> {
        let token_embeddings =
            token_inputs.mat_mul(&self.token_projection.broadcast_as([BATCH_SIZE, VOCAB_SIZE, MODEL_DIM]));
        let position_embeddings: Tensor<2, f32> = position_inputs.mat_mul(&self.position_projection);
        let mut x = token_embeddings.add_(&position_embeddings.broadcast_as([BATCH_SIZE, SEQ_LEN, MODEL_DIM]));

        let attn_input =
            x.layer_norm(&self.ln1_weight, Some(&self.ln1_bias), EPS, true);
        let q = attn_input.mat_mul(&self.w_q.broadcast_as([BATCH_SIZE, MODEL_DIM, MODEL_DIM]));
        let k = attn_input.mat_mul(&self.w_k.broadcast_as([BATCH_SIZE, MODEL_DIM, MODEL_DIM]));
        let v = attn_input.mat_mul(&self.w_v.broadcast_as([BATCH_SIZE, MODEL_DIM, MODEL_DIM]));

        let scores = q.mat_mul(&k.transpose(1, 2)) / (MODEL_DIM as f32).sqrt();
        let masked_scores = causal_mask.apply(&scores);
        let weights_exp = masked_scores.exp();
        let attention = weights_exp.div_(&weights_exp.sum_keepdim(2));
        let attention_output = attention
            .mat_mul(&v)
            .mat_mul(&self.w_o.broadcast_as([BATCH_SIZE, MODEL_DIM, MODEL_DIM]));
        x = x + attention_output;

        let ff_input = x.layer_norm(&self.ln2_weight, Some(&self.ln2_bias), EPS, true);
        let ff_hidden = ff_input
            .mat_mul(&self.w1.broadcast_as([BATCH_SIZE, MODEL_DIM, FF_DIM]))
            .add_(&self.b1)
            .relu();
        let ff_output = ff_hidden
            .mat_mul(&self.w2.broadcast_as([BATCH_SIZE, FF_DIM, MODEL_DIM]))
            .add_(&self.b2);
        x = x + ff_output;

        let output = x.layer_norm(&self.ln_out_weight, Some(&self.ln_out_bias), EPS, true);
        output.mat_mul(&self.lm_head.broadcast_as([BATCH_SIZE, MODEL_DIM, VOCAB_SIZE]))
    }

    async fn step(self, gradients: &Gradients, device: &Device) -> Self {
        Self {
            token_projection: sgd_step_2d::<VOCAB_SIZE, MODEL_DIM>(&self.token_projection, gradients, device).await,
            position_projection: sgd_step_2d::<SEQ_LEN, MODEL_DIM>(&self.position_projection, gradients, device).await,
            ln1_weight: sgd_step_1d::<MODEL_DIM>(&self.ln1_weight, gradients, device).await,
            ln1_bias: sgd_step_1d::<MODEL_DIM>(&self.ln1_bias, gradients, device).await,
            w_q: sgd_step_2d::<MODEL_DIM, MODEL_DIM>(&self.w_q, gradients, device).await,
            w_k: sgd_step_2d::<MODEL_DIM, MODEL_DIM>(&self.w_k, gradients, device).await,
            w_v: sgd_step_2d::<MODEL_DIM, MODEL_DIM>(&self.w_v, gradients, device).await,
            w_o: sgd_step_2d::<MODEL_DIM, MODEL_DIM>(&self.w_o, gradients, device).await,
            ln2_weight: sgd_step_1d::<MODEL_DIM>(&self.ln2_weight, gradients, device).await,
            ln2_bias: sgd_step_1d::<MODEL_DIM>(&self.ln2_bias, gradients, device).await,
            w1: sgd_step_2d::<MODEL_DIM, FF_DIM>(&self.w1, gradients, device).await,
            b1: sgd_step_1d::<FF_DIM>(&self.b1, gradients, device).await,
            w2: sgd_step_2d::<FF_DIM, MODEL_DIM>(&self.w2, gradients, device).await,
            b2: sgd_step_1d::<MODEL_DIM>(&self.b2, gradients, device).await,
            ln_out_weight: sgd_step_1d::<MODEL_DIM>(&self.ln_out_weight, gradients, device).await,
            ln_out_bias: sgd_step_1d::<MODEL_DIM>(&self.ln_out_bias, gradients, device).await,
            lm_head: sgd_step_2d::<MODEL_DIM, VOCAB_SIZE>(&self.lm_head, gradients, device).await,
        }
    }
}

#[tokio::main]
async fn main() {
    let device = Device::new().await.unwrap();

    let token_ids = training_sequences();
    let token_inputs: Tensor<3, f32> = Tensor::new(&device, &token_one_hot(&token_ids));
    let targets: Tensor<3, f32> = Tensor::new(&device, &next_token_one_hot(&token_ids));
    let position_inputs: Tensor<2, f32> = Tensor::new(&device, &position_one_hot());
    let causal_mask = AttentionMask::causal(&device, SEQ_LEN);

    let mut model = TinyTransformer::new(&device);

    for epoch in 0..EPOCHS {
        let logits = model.forward(&token_inputs, &position_inputs, &causal_mask);
        let error = &logits - &targets;
        let loss: Tensor<0, f32> = (&error * &error)
            .sum::<2>(2)
            .sum::<1>(1)
            .sum::<0>(0)
            / (BATCH_SIZE * SEQ_LEN * VOCAB_SIZE) as f32;

        let loss_value = loss.to_scalar().await.unwrap();
        let gradients = loss.backward().unwrap();
        model = model.step(&gradients, &device).await;

        if epoch % 20 == 0 || epoch + 1 == EPOCHS {
            println!("epoch {:>3}: loss={loss_value:.6}", epoch + 1);
        }
    }

    let logits = model.forward(&token_inputs, &position_inputs, &causal_mask);
    let predictions = argmax_last_dim(logits.to_vec3().await.unwrap());

    println!("training sequences:");
    for sequence in &token_ids {
        println!("  {sequence:?}");
    }
    println!("predicted next tokens:");
    for prediction in predictions {
        println!("  {prediction:?}");
    }
}

fn training_sequences() -> [[u32; SEQ_LEN]; BATCH_SIZE] {
    [
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 0],
        [4, 5, 0, 1],
        [5, 0, 1, 2],
    ]
}

fn token_one_hot(tokens: &[[u32; SEQ_LEN]; BATCH_SIZE]) -> [[[f32; VOCAB_SIZE]; SEQ_LEN]; BATCH_SIZE] {
    std::array::from_fn(|batch| {
        std::array::from_fn(|position| {
            let token = tokens[batch][position] as usize;
            std::array::from_fn(|vocab| if vocab == token { 1.0 } else { 0.0 })
        })
    })
}

fn next_token_one_hot(
    tokens: &[[u32; SEQ_LEN]; BATCH_SIZE],
) -> [[[f32; VOCAB_SIZE]; SEQ_LEN]; BATCH_SIZE] {
    std::array::from_fn(|batch| {
        std::array::from_fn(|position| {
            let token = ((tokens[batch][position] as usize) + 1) % VOCAB_SIZE;
            std::array::from_fn(|vocab| if vocab == token { 1.0 } else { 0.0 })
        })
    })
}

fn position_one_hot() -> [[f32; SEQ_LEN]; SEQ_LEN] {
    std::array::from_fn(|position| {
        std::array::from_fn(|column| if column == position { 1.0 } else { 0.0 })
    })
}

fn random_matrix<const ROWS: usize, const COLS: usize>(
    device: &Device,
    rng: &mut StdRng,
    scale: f32,
) -> Tensor<2, f32> {
    let data: [[f32; COLS]; ROWS] = std::array::from_fn(|_| {
        std::array::from_fn(|_| rng.random_range(-scale..scale))
    });
    Tensor::new(device, &data)
}

fn ones<const LEN: usize>(device: &Device) -> Tensor<1, f32> {
    Tensor::new(device, &[1.0; LEN])
}

fn zeros<const LEN: usize>(device: &Device) -> Tensor<1, f32> {
    Tensor::new(device, &[0.0; LEN])
}

async fn sgd_step_1d<const LEN: usize>(
    parameter: &Tensor<1, f32>,
    gradients: &Gradients,
    device: &Device,
) -> Tensor<1, f32> {
    let gradient = gradients.get(parameter).unwrap();
    let next = parameter - &(gradient * LEARNING_RATE);
    let host = next.to_vec1().await.unwrap();
    let host: [f32; LEN] = host.try_into().unwrap();
    Tensor::new(device, &host)
}

async fn sgd_step_2d<const ROWS: usize, const COLS: usize>(
    parameter: &Tensor<2, f32>,
    gradients: &Gradients,
    device: &Device,
) -> Tensor<2, f32> {
    let gradient = gradients.get(parameter).unwrap();
    let next = parameter - &(gradient * LEARNING_RATE);
    let host = next.to_vec2().await.unwrap();
    let host: [[f32; COLS]; ROWS] =
        std::array::from_fn(|row| std::array::from_fn(|col| host[row][col]));
    Tensor::new(device, &host)
}

fn argmax_last_dim(logits: Vec<Vec<Vec<f32>>>) -> Vec<Vec<usize>> {
    logits
        .into_iter()
        .map(|sequence| {
            sequence
                .into_iter()
                .map(|token_logits| {
                    token_logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, left), (_, right)| left.total_cmp(right))
                        .map(|(index, _)| index)
                        .unwrap()
                })
                .collect()
        })
        .collect()
}
