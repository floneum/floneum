use kalosm::language::*;

#[tokio::main]
async fn main() {
    let model = Llama::new_chat().await.unwrap();
    let mut chat = model
        .chat()
        .with_system_prompt("The assistant will act like a pirate");

    loop {
        chat(&prompt_input("\n> ").unwrap())
            .to_std_out()
            .await
            .unwrap();
    }
}
