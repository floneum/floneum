use kalosm::language::*;

#[tokio::main]
async fn main() {
    let model = Llama::builder()
        .with_source(LlamaSource::qwen_2_5_0_5b_instruct())
        .build()
        .await
        .unwrap();
    let mut chat = Chat::builder(model)
        .with_system_prompt("You will act like a pirate")
        .build();

    loop {
        chat.add_message(prompt_input("\n> ").unwrap())
            .to_std_out()
            .await
            .unwrap();
    }
}
