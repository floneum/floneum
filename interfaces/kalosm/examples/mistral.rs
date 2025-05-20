use kalosm::language::*;

#[tokio::main]
async fn main() {
    let llm = Llama::builder()
        .with_source(LlamaSource::mistral_7b())
        .build()
        .await
        .unwrap();
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    print!("{prompt}");

    llm(prompt).to_std_out().await.unwrap();
}
