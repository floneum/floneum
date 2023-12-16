use kalosm::{language::*, *};

#[tokio::main]
async fn main() {
    let mut model = Llama::builder()
        .with_source(LlamaSource::mistral_7b_instruct_2())
        .build()
        .unwrap();
    let mut chat = Chat::builder(&mut model)
        .with_system_prompt("The assistant will act like a pirate")
        .build();

    loop {
        chat.add_message(prompt_input("\n> ").unwrap())
            .await
            .unwrap()
            .to_std_out()
            .await
            .unwrap();
    }
}
