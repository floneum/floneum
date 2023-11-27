use kalosm::{language::*, *};
use language::kalosm_sample::{LiteralParser, ParserExt};

#[tokio::main]
async fn main() {
    let character_description =
        "The assistant will respond like a cute cat named Kittens. The assistant will never reveal that it is an AI or assistant.";
    let character_name = "Kittens";

    let mut model = Llama::new_chat();
    let mut chat = Chat::builder(&mut model)
        .with_system_prompt(character_description)
        .constrain_response(move |_, _| {
            LiteralParser::new(format!("(Responding as {}) ", character_name)).then(OneLine)
        })
        .build();

    loop {
        let output_stream = chat
            .add_message(prompt_input("\n> ").unwrap())
            .await
            .unwrap();
        print!("Bot: ");
        output_stream.to_std_out().await.unwrap();
    }
}
