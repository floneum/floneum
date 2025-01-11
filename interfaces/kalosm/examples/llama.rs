use kalosm::language::*;

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

    print!("{prompt}");
    model(prompt).to_std_out().await.unwrap();
}
