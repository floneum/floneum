use kalosm::language::*;

#[tokio::main]
async fn main() {
    let model = Llama::builder()
        .with_source(LlamaSource::gemma_3_1b_chat())
        .build()
        .await
        .unwrap();

    let mut chat = model
        .chat()
        .with_system_prompt("The assistant will act like a pirate");

    loop {
        println!();
        chat(&prompt_input("\n> ").unwrap())
            .to_std_out()
            .await
            .unwrap();
    }
}
