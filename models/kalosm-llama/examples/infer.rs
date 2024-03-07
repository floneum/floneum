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
    let mut result = model.stream_text(prompt).await.unwrap();

    print!("{prompt}");
    while let Some(token) = result.next().await {
        print!("{token}");
        std::io::stdout().flush().unwrap();
    }
}
