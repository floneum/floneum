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
                "https://preview.redd.it/bgw66bknw32f1.png?width=1080&crop=smart&auto=webp&s=128c68de9c9301e80d73f94cac07cc2970f28f34",
            ),
            MediaType::Image,
        ),
        "Describe this image.",
    ));
    response.to_std_out().await.unwrap();
    response.await.unwrap();
    println!("\n");
}
