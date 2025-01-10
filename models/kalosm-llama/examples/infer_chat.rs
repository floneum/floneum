use kalosm_llama::prelude::*;
use kalosm_streams::text_stream::TextStream;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let model = Llama::builder()
        .with_source(LlamaSource::llama_3_2_3b_chat())
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
