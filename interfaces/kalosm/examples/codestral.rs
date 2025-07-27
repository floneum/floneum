use kalosm::language::*;

#[tokio::main]
async fn main() {
    let model = Llama::builder()
        .with_source(LlamaSource::codestral_22b())
        .build()
        .await
        .unwrap();

    let mut chat = model.chat();

    let mut stream = chat(&"Finish this code: fn main() { println");

    stream.to_std_out().await.unwrap();

    stream.await.unwrap();
}
