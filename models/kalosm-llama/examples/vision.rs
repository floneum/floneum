use kalosm_llama::prelude::*;
use kalosm_streams::text_stream::TextStream;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let model = Llama::builder()
        .with_source(
            LlamaSource::new(kalosm_model_types::FileSource::Local(
                "/Users/evanalmloff/Desktop/Github/candle/qwen_2_5_3b_f16.gguf".into(),
            ))
            .with_tokenizer(
                kalosm_model_types::FileSource::HuggingFace {
                    model_id: "Qwen/Qwen2.5-VL-3B-Instruct".into(),
                    revision: "main".into(),
                    file: "tokenizer.json".into(),
                },
            ),
        )
        .build()
        .await
        .unwrap();

    println!("First message\n");

    let mut chat = model.chat();
    loop {
        let mut response = chat(&prompt_input("> ").unwrap());
        response.to_std_out().await.unwrap();
        println!("\n");
    }
}
