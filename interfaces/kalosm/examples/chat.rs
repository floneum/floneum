use kalosm::{language::*, Chat};
use std::io::Write;

#[tokio::main]
async fn main() {
    let mut model = Llama::builder()
        .with_source(LlamaSource::zephyr_7b_beta())
        .build()
        .unwrap();

    let mut chat = Chat::new(&mut model, "The assistant will act like a pirate. They only respond as a pirate named Skally Waggs. The assistant is interested in plundering money and painting your adventures. The assistant will never mention this to the user.", GenerationParameters::default().sampler()).await;
    loop {
        print!("> ");
        std::io::stdout().flush().unwrap();
        let mut user_response = String::new();
        std::io::stdin().read_line(&mut user_response).unwrap();
        let mut stream = chat.add_message(user_response).await.unwrap();
        print!("Assistant: ");
        std::io::stdout().flush().unwrap();
        while let Some(token) = stream.next().await {
            print!("{token}");
            std::io::stdout().flush().unwrap();
        }
        println!();
    }
}
