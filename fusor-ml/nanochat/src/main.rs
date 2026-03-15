mod config;
mod data;
mod model;

use data::{
    ChatDataset, Tokenizer, autoregressive_context, one_hot, position_one_hot, windows_to_inputs,
    windows_to_targets,
};
use fusor_core::{Device, Tensor, cache::AttentionMask};
use model::NanoChatModel;
use rand::{SeedableRng, rngs::StdRng};

use crate::config::{BATCH_SIZE, BLOCK_SIZE, EOT_TOKEN, LOG_EVERY, SAMPLE_TOKENS, TRAIN_STEPS, VOCAB_SIZE};

fn main() {
    pollster::block_on(async {
        let tokenizer = Tokenizer;
        let dataset = ChatDataset::from_tsv(include_str!("../chat.txt"), &tokenizer);
        assert!(
            dataset.num_docs() >= 80,
            "embedded chat dataset is unexpectedly small: {} docs",
            dataset.num_docs()
        );
        assert!(
            dataset.max_tokens_per_example() <= BLOCK_SIZE + 1,
            "max example length {} exceeds block size {}",
            dataset.max_tokens_per_example(),
            BLOCK_SIZE + 1
        );
        let mut rng = StdRng::seed_from_u64(1337);

        let device = Device::new().await.unwrap();
        let position_inputs: Tensor<2, f32> = Tensor::new(&device, &position_one_hot());
        let causal_mask = AttentionMask::causal(&device, BLOCK_SIZE);
        let mut model = NanoChatModel::new(&device, &mut rng);

        println!(
            "docs={} tokens={} vocab={} params={} max_example_tokens={}",
            dataset.num_docs(),
            dataset.num_tokens(),
            VOCAB_SIZE,
            model.num_parameters(),
            dataset.max_tokens_per_example(),
        );

        for step in 0..TRAIN_STEPS {
            let batch = dataset.sample_batch(&mut rng);
            let token_inputs: Tensor<3, f32> =
                Tensor::new(&device, &windows_to_inputs(&batch.windows));
            let targets: Tensor<3, f32> =
                Tensor::new(&device, &windows_to_targets(&batch.windows));
            let mask: Tensor<2, f32> = Tensor::new(&device, &batch.mask);

            let logits = model.forward::<BATCH_SIZE>(&token_inputs, &position_inputs, &causal_mask);
            let loss = masked_cross_entropy::<BATCH_SIZE>(&logits, &targets, &mask, batch.valid_tokens);

            let gradients = loss.backward().unwrap();
            let should_log = step % LOG_EVERY == 0 || step + 1 == TRAIN_STEPS;
            let loss_value = if should_log {
                Some(loss.to_scalar().await.unwrap())
            } else {
                None
            };
            model = model.step(&gradients);

            if let Some(loss_value) = loss_value {
                println!("step {:>4} | loss={loss_value:.6}", step + 1);
            }
        }

        println!("\n--- training prompt eval ---");
        for example in dataset.examples().iter().take(8) {
            let reply = generate_reply(
                &model,
                &device,
                &position_inputs,
                &causal_mask,
                &tokenizer,
                example.user(),
            )
            .await;
            println!("user: {}", example.user());
            println!("expected: {}", example.assistant());
            println!("assistant: {reply}\n");
        }
    });
}

async fn generate_reply(
    model: &NanoChatModel,
    device: &Device,
    position_inputs: &Tensor<2, f32>,
    causal_mask: &AttentionMask<f32>,
    tokenizer: &Tokenizer,
    prompt: &str,
) -> String {
    let mut tokens = tokenizer.encode_chat_prompt(prompt);
    let prompt_len = tokens.len();

    for _ in 0..SAMPLE_TOKENS {
        let (context, last_index) = autoregressive_context(&tokens);
        let batch = [context];
        let token_inputs: Tensor<3, f32> = Tensor::new(device, &one_hot(&batch));
        let logits = model.forward::<1>(&token_inputs, position_inputs, causal_mask);
        let logits = logits.to_vec3().await.unwrap();
        let next = sample_from_logits(&logits[0][last_index]);
        tokens.push(next);
        if next == EOT_TOKEN {
            break;
        }
    }

    tokenizer.decode_assistant_reply(prompt_len, &tokens)
}

fn sample_from_logits(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index as u32)
        .unwrap()
}

fn masked_cross_entropy<const B: usize>(
    logits: &Tensor<3, f32>,
    targets: &Tensor<3, f32>,
    mask: &Tensor<2, f32>,
    valid_tokens: f32,
) -> Tensor<0, f32> {
    let log_norm = logits.exp().sum_keepdim::<2>(2).log();
    let log_probs = logits.sub_(&log_norm.broadcast_as([B, BLOCK_SIZE, VOCAB_SIZE]));
    let token_nll = -((targets * &log_probs).sum::<2>(2) * mask);
    token_nll.sum::<1>(1).sum::<0>(0) / valid_tokens.max(1.0)
}
