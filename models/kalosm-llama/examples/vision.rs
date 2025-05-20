use kalosm_llama::prelude::*;
use kalosm_streams::text_stream::TextStream;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let model = Llama::builder()
        .with_source(LlamaSource::qwen_2_5_3b_vl_chat_f16())
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
