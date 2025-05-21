use kalosm_llama::prelude::*;
use kalosm_streams::text_stream::TextStream;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let model = Llama::builder()
        .with_source(LlamaSource::qwen_2_5_3b_vl_chat_q4())
        .build()
        .await
        .unwrap();

    let mut chat = model.chat();
    let mut response = chat(&(
        MediaChunk::new(
            MediaSource::url(
                "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            ),
            MediaType::Image,
        )
        .with_hints(MediaHints::new().with_max_pixels(512)),
        "Describe this image.",
    ));
    response.to_std_out().await.unwrap();
    response.await.unwrap();
    println!("\n");

    loop {
        chat(&prompt_input("\n> ").unwrap())
            .to_std_out()
            .await
            .unwrap();
    }
}
