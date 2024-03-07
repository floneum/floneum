use kalosm::language::*;
use std::io::Write;

#[tokio::main]
async fn main() {
    let model = Llama::builder()
        .with_source(LlamaSource::zephyr_7b_beta())
        .build()
        .await
        .unwrap();
    let prompt = "<|system|>

</s>
<|user|>
What is your favorite story from your adventures?</s>
<|assistant|>";
    let mut result = model
        .stream_text(prompt)
        .with_max_length(1000)
        .await
        .unwrap();

    print!("{prompt}");
    while let Some(token) = result.next().await {
        print!("{token}");
        std::io::stdout().flush().unwrap();
    }
}
