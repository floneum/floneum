use kalosm::language::*;
use kalosm_language::kalosm_llama::LlamaChatSession;

#[tokio::main]
async fn main() {
    let model = Llama::new_chat().await.unwrap();
    let save_path = std::path::PathBuf::from("./chat.llama");
    let mut chat = model.chat();
    if let Some(old_session) = std::fs::read(&save_path).ok().and_then(|bytes|LlamaChatSession::from_bytes(&bytes).ok()) {
        chat = chat.with_session(old_session);
    } else {
        chat = chat.with_system_prompt("The assistant will act like a pirate. They only respond as a pirate named Skally Waggs. The assistant is interested in plundering money and painting your adventures. The assistant will never mention this to the user.");
    }
    

    for _ in 0..2 {
        let mut output_stream = chat.add_message(prompt_input("\n> ").unwrap());
        print!("Bot: ");
        output_stream.to_std_out().await.unwrap();
    }

    let bytes = chat.session().unwrap().to_bytes().unwrap();
    std::fs::write(save_path, bytes).unwrap();
}
