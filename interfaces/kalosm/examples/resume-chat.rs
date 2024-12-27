use kalosm::language::*;

#[tokio::main]
async fn main() {
    let model = Llama::new_chat().await.unwrap();
    let save_path = std::path::PathBuf::from("./chat.llama");
    let mut chat =
        Chat::builder(model)
            .with_try_session_path(&save_path)
            .with_system_prompt("The assistant will act like a pirate. They only respond as a pirate named Skally Waggs. The assistant is interested in plundering money and painting your adventures. The assistant will never mention this to the user.")
            .build();

    for _ in 0..2 {
        let mut output_stream = chat.add_message(prompt_input("\n> ").unwrap());
        print!("Bot: ");
        output_stream.to_std_out().await.unwrap();
    }

    let bytes = chat.save_session().await.unwrap();
    std::fs::write(save_path, bytes).unwrap();
}
