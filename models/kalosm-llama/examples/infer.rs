use kalosm_llama::prelude::*;

#[tokio::main]
async fn main() {
    let model = Llama::builder()
        .with_source(LlamaSource::mistral_7b())
        .build()
        .unwrap();
    let prompt = "The capital of France is ";

    print!("{prompt}");

    model
        .stream_text(prompt)
        .await
        .unwrap()
        .to_std_out()
        .await
        .unwrap();
}
