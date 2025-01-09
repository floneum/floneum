use std::io::Write;

use kalosm_llama::prelude::*;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let model = Llama::builder()
        .with_source(LlamaSource::llama_3_2_3b_chat())
        .build()
        .await
        .unwrap();

    println!("First message\n");

    let mut chat = Chat::new(model);
    loop {
        let mut response = chat.add_message(prompt_input("> ").unwrap());
        while let Some(text) = response.next().await {
            print!("{text}");
            _ = std::io::stdout().flush();
        }
        let all_text = response.await.unwrap();
        println!("{all_text}");
        println!("\n");
    }
}
