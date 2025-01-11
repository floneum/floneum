//! You can have multiple chat instances with the same model.

use kalosm::language::*;
use std::io::Write;

#[tokio::main]
async fn main() {
    let model = Llama::new_chat().await.unwrap();
    let mut agent1 = model
        .chat()
        .with_system_prompt("The assistant will act like a pirate.");
    let mut agent2 = model
        .chat()
        .with_system_prompt("The assistant is curious and will ask questions about the world.");

    let mut response = String::from("Is there anything you want to know about the world?");
    loop {
        println!("User:");
        let mut stream = agent2(&response);
        let mut user_question = String::new();
        while let Some(token) = stream.next().await {
            print!("{token}");
            std::io::stdout().flush().unwrap();
            user_question += &token;
        }
        println!();

        println!("Assistant:");
        let mut stream = agent1(&user_question);
        response.clear();
        while let Some(token) = stream.next().await {
            print!("{token}");
            std::io::stdout().flush().unwrap();
            response += &token;
        }
        println!();
    }
}
