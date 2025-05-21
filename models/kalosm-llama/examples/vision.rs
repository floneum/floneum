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
    let mut response = chat(&(
        MediaChunk::new(
            MediaSource::url(
                "https://avatars.githubusercontent.com/u/76850177?v=4",
            ),
            MediaType::Image,
        ),
        "Describe this image.",
    ));
    response.to_std_out().await.unwrap();
    response.await.unwrap();
    println!("\n");
}
