use kalosm_llama::prelude::*;
use kalosm_streams::text_stream::TextStream;

#[tokio::main]
async fn main() {
    let model = Llama::builder()
        .with_source(LlamaSource::llama_8b())
        .build()
        .await
        .unwrap();

    let mut story = model(&"Once upon a time there was a penguin named Peng.");

    story.to_std_out().await.unwrap();
}
