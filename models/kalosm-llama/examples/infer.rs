use kalosm::language::*;

#[tokio::main]
async fn main() {
    let model = Llama::builder()
        .with_source(LlamaSource::qwen_2_5_0_5b_instruct())
        .build()
        .await
        .unwrap();

    model
        .stream_text("The capital of France is ")
        .with_max_length(100)
        .await
        .unwrap()
        .to_std_out()
        .await
        .unwrap();
}
