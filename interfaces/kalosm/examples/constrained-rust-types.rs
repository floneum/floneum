use kalosm::language::*;

#[tokio::main]
async fn main() {
    let llm = Llama::builder()
        .with_source(LlamaSource::tiny_llama_1_1b())
        .build()
        .await
        .unwrap();
    let prompt = "An array of realistic single word reddit user names: ";

    let validator = <[Word<1, 10>; 10] as HasParser>::new_parser();
    let words = llm.stream_structured_text(prompt, validator).await.unwrap();

    println!("\n{:#?}", words.result().await);
}
