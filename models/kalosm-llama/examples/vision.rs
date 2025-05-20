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
                "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            ),
            MediaType::Image,
        ),
        "Describe the image in detail.",
    ));
    response.to_std_out().await.unwrap();
    response.await.unwrap();
    println!("\n");
}
