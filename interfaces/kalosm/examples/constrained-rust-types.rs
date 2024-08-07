#![allow(unused)]
use kalosm::language::*;

// You can derive an efficient parser for your struct with the `Parse` trait
#[derive(Schema, Parse, Clone, Debug)]
struct Account {
    // User names only contain alphanumeric characters and spaces
    #[parse(with = StringParser::new(1..=20).alphanumeric_with_spaces())]
    username: String,
    // The age must be between 1 and 100
    #[parse(with = U8Parser::new().with_range(1..=100))]
    age: u8,
}

type List = Box<[Account; 100]>;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Download phi-3 or get the model from the cache
    let llm = Llama::phi_3().await?;

    // Create a task that generates a list of accounts
    let task = Task::builder("You generate realistic JSON placeholders")
        .with_constraints(<List as Parse>::new_parser())
        .build();
    let prompt = format!("Generate JSON in this format: {}", List::schema());
    println!("{prompt}");
    let mut stream = task.run(prompt, &llm);

    // Stream the output to the console
    stream.to_std_out().await?;

    // And get the typed result once the stream is finished
    let accounts = stream.await?;
    println!("{:#?}", accounts);

    Ok(())
}
