use kalosm_llama::prelude::*;
use std::io::Write;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let model = Llama::builder()
        .with_source(
            LlamaSource::new(kalosm_model_types::FileSource::HuggingFace {
                model_id: "unsloth/SmolLM2-135M-Instruct-GGUF".into(),
                revision: "main".into(),
                file: "SmolLM2-135M-Instruct-Q4_K_M.gguf".into(),
            })
            .with_tokenizer(kalosm_model_types::FileSource::HuggingFace {
                model_id: "HuggingFaceTB/SmolLM2-135M-Instruct".into(),
                revision: "main".into(),
                file: "tokenizer.json".into(),
            }),
        )
        .build()
        .await
        .unwrap();

    let mut story = model("Once upon a time there was a penguin named Peng.");

    let start = std::time::Instant::now();
    let mut tokens = 0;
    while let Some(token) = story.next().await {
        print!("{}", token);
        std::io::stdout().flush().unwrap();
        tokens += 1;
        let elapsed = start.elapsed();
        if elapsed.as_secs() > 0 {
            let tokens_per_second = tokens as f64 / elapsed.as_secs_f64();
            println!(
                "\n{} tokens in {:.2} seconds ({:.2} tokens/second)",
                tokens,
                elapsed.as_secs_f64(),
                tokens_per_second
            );
        }
    }
}
