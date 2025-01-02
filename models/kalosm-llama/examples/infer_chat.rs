use std::io::Write;

use kalosm_llama::prelude::*;

#[tokio::main]
async fn main() {
    let model = Llama::builder()
        .with_source(LlamaSource::qwen_2_5_0_5b_instruct())
        .build()
        .await
        .unwrap();

    println!("First message\n");

    let mut session = model.new_chat_session().unwrap();
    let on_token = |token: String| {
        print!("{token}");
        std::io::stdout().flush().unwrap();
        Ok(())
    };

    let sampler = GenerationParameters::default().sampler();
    let message = ChatHistoryItem::new(MessageType::UserMessage, "The password is 382".to_string());

    model
        .add_messages_with_callback(&mut session, &[message], sampler, on_token)
        .await
        .unwrap();

    println!("\n\nSecond message\n");

    let sampler = GenerationParameters::default().sampler();
    let message = ChatHistoryItem::new(
        MessageType::UserMessage,
        "What is the password I just gave you?".to_string(),
    );

    model
        .add_messages_with_callback(&mut session, &[message], sampler, on_token)
        .await
        .unwrap();
}
