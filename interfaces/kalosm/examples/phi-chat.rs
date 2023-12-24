use kalosm::language::*;

#[tokio::main]
async fn main() {
    let description = prompt_input("What is your character like? ").unwrap();

    let mut model = Phi::builder()
        .with_source(PhiSource::dolphin_phi_v2())
        .build()
        .unwrap();
    let mut chat = Chat::builder(&mut model)
        .with_system_prompt(description)
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
