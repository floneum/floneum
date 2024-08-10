#![allow(unused)]
use kalosm::language::*;

// You can derive an efficient parser for your struct with the `Parse` trait
#[derive(Schema, Parse, Clone, Debug)]
struct Account {
    /// A summary of the account holder
    #[parse(pattern = r"[a-zA-Z,.?!\d ]{1,80}")]
    summary: String,
    /// The user name of the account holder. This may be the full name of the user or a pseudonym
    #[parse(pattern = "[a-zA-Z ]{1,20}")]
    username: String,
    /// The age of the account holder
    #[parse(range = 1..=100)]
    age: u8,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Download default chat model or get the model from the cache
    let llm = Llama::phi_3().await?;

    // Create a task that generates a list of accounts
    let task = Task::builder_for::<Account>(
        "You generate accounts based on a description of the account holder",
    )
    .with_example(
        "John Doe is a 25 year old accountant who works at ABC Inc.",
        r#"{ "summary": "John Doe is 25", "username": "John Doe", "age": 25 }"#,
    )
    .with_example(
        "A 57 years old woman named Candace.",
        r#"{ "summary": "Candace is 57", "username": "Candace", "age": 57 }"#,
    )
    .with_example(
        "Anderson is a programmer who works at Google, but he goes by Mr Robot. He is 71 years old.",
        r#"{ "summary": "Mr Robot is 71", "username": "Mr Robot", "age": 71 }"#,
    )
    .build();

    loop {
        let prompt = prompt_input("Describe an account holder: ").unwrap();
        let mut stream = task.run(prompt, &llm);

        // Stream the output to the console
        stream.to_std_out().await?;

        // And get the typed result once the stream is finished
        let accounts = stream.await?;
        println!("\n{:#?}", accounts);
    }

    Ok(())
}
