use kalosm::language::*;
use std::io::Write;

#[tokio::main]
async fn main() {
    let mut model = Llama::builder()
        .with_source(LlamaSource::zephyr_7b_beta())
        .build()
        .unwrap();
    let prompt = "<|system|>
    </s>
    <|user|>
    who are you</s>
    <|assistant|>";
    // let prompt = "What is the capital of france?";
    let mut result = model.stream_text(prompt).await.unwrap();

    print!("{prompt}");
    while let Some(token) = result.next().await {
        print!("{token}");
        std::io::stdout().flush().unwrap();
    }
}
