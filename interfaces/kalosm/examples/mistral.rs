use std::io::Write;

use kalosm::language::*;

#[tokio::main]
async fn main() {
    let llm = Llama::builder()
        .with_source(LlamaSource::mistral_7b())
        .build()
        .await
        .unwrap();
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    print!("{}", prompt);

    let stream = llm.stream_text(prompt).with_max_length(300).await.unwrap();

    let mut sentences = stream.words();
    while let Some(text) = sentences.next().await {
        print!("{}", text);
        std::io::stdout().flush().unwrap();
    }
}
