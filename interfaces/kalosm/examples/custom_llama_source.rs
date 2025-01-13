use kalosm::language::*;

#[tokio::main]
async fn main() {
    let llm = Llama::builder()
        // To use a custom model, you can set the LlamaSource to a custom model
        .with_source(LlamaSource::new(
            // Llama source takes a gguf file to load the model and tokenizer from
            FileSource::HuggingFace {
                model_id: "QuantFactory/SmolLM-135M-GGUF".to_string(),
                revision: "main".to_string(),
                file: "SmolLM-135M.Q4_K_M.gguf".to_string(),
            },
        ))
        // If you are using a custom chat model, you also need to set the chat markers with the with_chat_markers method
        .build()
        .await
        .unwrap();
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    print!("{}", prompt);

    llm(prompt).to_std_out().await.unwrap();
}
