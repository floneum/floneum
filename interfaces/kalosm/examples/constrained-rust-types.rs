#![allow(unused)]
use kalosm::language::*;

// You can derive an efficient parser for your struct with the `Parse` trait
#[derive(Parse, Clone, Debug)]
struct Account {
    username: String,
    age: u8,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Download phi-3 or get the model from the cache
    let llm = Llama::phi_3().await?;

    // Create a task that generates a list of accounts
    let task = Task::builder("You generate realistic JSON placeholders")
        .with_constraints(<[Account; 10] as Parse>::new_parser())
        .build();
    let prompt =
        "Generate a list of 10 accounts with random realistic usernames and ages in JSON format";
    let mut stream = task.run(prompt, &llm);

    // Stream the output to the console
    stream.to_std_out().await?;

    // And get the typed result once the stream is finished
    let accounts = stream.await?;
    println!("{:#?}", accounts);

    Ok(())
}
