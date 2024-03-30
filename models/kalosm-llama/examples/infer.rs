use kalosm_common::ModelLoadingProgress;
use kalosm_llama::prelude::*;
use std::io::Write;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let model = Llama::builder()
        .with_source(LlamaSource::mistral_7b())
        .build_with_loading_handler(|loading| match loading {
            ModelLoadingProgress::Downloading { source, progress } => {
                println!("Downloading {} at {:0.2}%", source, progress * 100.);
            }
            ModelLoadingProgress::Loading { progress } => {
                println!("Loading {:0.2}%", progress * 100.);
            }
        })
        .await
        .unwrap();
    let prompt = "The capital of France is ";
    let mut result = model
        .stream_text(prompt)
        .with_max_length(100)
        .await
        .unwrap();

    let start_time = std::time::Instant::now();
    let mut tokens = 0;
    print!("{prompt}");
    while let Some(token) = result.next().await {
        tokens += 1;
        print!("{token}");
        std::io::stdout().flush().unwrap();
    }
    let elapsed = start_time.elapsed();
    println!("\n\nGenerated {} tokens in {:?}", tokens, elapsed);
    println!(
        "Tokens per second: {:.2}",
        tokens as f64 / elapsed.as_secs_f64()
    );
}
