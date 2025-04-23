use kalosm_llama::prelude::*;
use kalosm_streams::text_stream::TextStream;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let model = Llama::builder()
        .with_source(
            LlamaSource::new(kalosm_model_types::FileSource::HuggingFace {
                model_id: "unsloth/SmolLM2-135M-Instruct-GGUF".into(),
                revision: "main".into(),
                file: "SmolLM2-135M-Instruct-Q4_K_M.gguf".into(),
            })
            .with_tokenizer(kalosm_model_types::FileSource::HuggingFace {
                model_id: "HuggingFaceTB/SmolLM2-135M-Instruct".into(),
                revision: "main".into(),
                file: "tokenizer.json".into(),
            }),
        )
        .build()
        .await
        .unwrap();

    let mut story = model("Once upon a time there was a penguin named Peng.");

    story.to_std_out().await.unwrap();
}
