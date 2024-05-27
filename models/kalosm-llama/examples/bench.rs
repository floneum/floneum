use std::sync::{Arc, Mutex};

use kalosm_common::ModelLoadingProgress;
use kalosm_language_model::{GenerationParameters, SyncModel, SyncModelExt};
use kalosm_llama::*;

#[tokio::main]
async fn main() {
    fn progress(_: ModelLoadingProgress) {}

    #[inline(never)]
    async fn load_small() {
        let model = LlamaModel::from_builder(
            Llama::builder().with_source(LlamaSource::llama_8b()),
            progress,
        )
        .await
        .unwrap();
        let prompt = "Hello world";

        let tokens = model.tokenizer().encode(prompt, true).unwrap().len();
        for _ in 0..100 {
            let start_time = std::time::Instant::now();
            let mut session = model.new_session().unwrap();
            model.feed_text(&mut session, prompt).unwrap();
            let elapsed = start_time.elapsed();
            println!("\n\nLoaded {} tokens in {:?}", tokens, elapsed);
            println!(
                "Tokens per second: {:.2}",
                tokens as f64 / elapsed.as_secs_f64()
            );
        }
    }

    #[inline(never)]
    async fn load_large() {
        let model = LlamaModel::from_builder(
            Llama::builder().with_source(LlamaSource::llama_8b()),
            progress,
        )
        .await
        .unwrap();
        let prompt = "Hello world".repeat(10);

        let tokens = model.tokenizer().encode(&prompt, true).unwrap().len();
        for _ in 0..100 {
            let start_time = std::time::Instant::now();
            let mut session = model.new_session().unwrap();
            model.feed_text(&mut session, &prompt).unwrap();
            let elapsed = start_time.elapsed();
            println!("\n\nLoaded {} tokens in {:?}", tokens, elapsed);
            println!(
                "Tokens per second: {:.2}",
                tokens as f64 / elapsed.as_secs_f64()
            );
        }
    }

    #[inline(never)]
    async fn generate() {
        let model = LlamaModel::from_builder(
            Llama::builder().with_source(LlamaSource::llama_8b()),
            progress,
        )
        .await
        .unwrap();
        let prompt = "Hello world";

        for _ in 0..100 {
            let mut session = model.new_session().unwrap();
            let start_time = std::time::Instant::now();
            let tokens = 100;
            model
                .stream_text_with_sampler(
                    &mut session,
                    prompt,
                    Some(100),
                    None,
                    Arc::new(Mutex::new(GenerationParameters::default().sampler())),
                    |_| Ok(kalosm_language_model::ModelFeedback::Continue),
                )
                .unwrap();
            let elapsed = start_time.elapsed();
            println!("\n\nGenerated {} tokens in {:?}", tokens, elapsed);
            println!(
                "Tokens per second: {:.2}",
                tokens as f64 / elapsed.as_secs_f64()
            );
        }
    }

    load_small().await;
    load_large().await;
    generate().await;
}
