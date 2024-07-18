#![allow(unused)]
use std::sync::Arc;
use kalosm::language::*;

#[tokio::main]
async fn main() {
    // You can derive an efficient parser for your struct with the `Parse` trait
    #[derive(Parse, Clone)]
    pub enum Response {
        Do(String),
        Say(String),
    }

    // Create a parser and warp it in Arc so it implements Clone
    let parser = Arc::new(Response::new_parser());

    // Create a chat session with the default chat model with the parser as constraints
    let model = Llama::new_chat().await.unwrap();
    let mut chat = Chat::builder(model)
        .with_constraints(move |_history| parser.clone())
        .with_system_prompt("The assistant will act like a pirate. You will respond with either something you do or something you say. Respond with JSON in the format { \"type\": \"Say\", \"data\": \"hello\" } or { \"type\": \"Do\", \"data\": \"run away\" }")
        .build();

    // Then chat with the session
    loop {
        chat.add_message(prompt_input("\n> ").unwrap())
            .to_std_out()
            .await
            .unwrap();
    }
}
