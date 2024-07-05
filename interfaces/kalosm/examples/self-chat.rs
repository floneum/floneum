//! You can have multiple chat instances with the same model.

use kalosm::language::*;
use std::io::Write;

#[tokio::main]
async fn main() {
    let model = Llama::new_chat().await.unwrap();
    let mut agent1 = Chat::builder(model.clone())
        .with_system_prompt("The assistant will act like a pirate.")
        .build();
    let mut agent2 = Chat::builder(model)
        .with_system_prompt("The assistant is curious and will ask questions about the world.")
        .build();

    let mut response = String::from("Is there anything you want to know about the world?");
    loop {
        println!("User:");
        let mut stream = agent2.add_message(&response);
        let mut user_question = String::new();
        while let Some(token) = stream.next().await {
            print!("{token}");
            std::io::stdout().flush().unwrap();
            user_question += &token;
        }
        println!();

        println!("Assistant:");
        let mut stream = agent1.add_message(user_question);
        response.clear();
        while let Some(token) = stream.next().await {
            print!("{token}");
            std::io::stdout().flush().unwrap();
            response += &token;
        }
        println!();
    }
}
