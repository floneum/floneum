use kalosm::language::*;
use std::io::Write;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let llm = Llama::new().await.unwrap();
    let prompt = r#"```json
[
{ name: "bob", description: "An adorable cute cat", color: "black", size: "small", diet: "carnivore", breeds: ["Persian", "Maine Coon"] },
"#;

    {
        println!("# with constraints");
        print!("{}", prompt);

        let regex = r#"(\{ name: "\w+", description: "[\w ]+", color: "\w+", size: "\w+", diet: "\w+", breeds: \[("[\w ]+", )*"[\w ]+"\] \},){4}\n\]"#;
        let validator = RegexParser::new(regex).unwrap();
        let stream = llm.stream_structured_text(prompt, validator).await.unwrap();

        time_stream(stream).await;
    }

    {
        println!("# with constraints 2");
        print!("{}", prompt);

        let validator = LiteralParser::new("{ name: ")
            .then(WordParser::new())
            .then(LiteralParser::new(", description: "))
            .then(StringParser::new(1..=50))
            .then(LiteralParser::new(", color: "))
            .then(WordParser::new())
            .then(LiteralParser::new(", size: "))
            .then(WordParser::new())
            .then(LiteralParser::new(", diet: "))
            .then(WordParser::new())
            .then(LiteralParser::new(", breeds: "))
            .then(RegexParser::new(r#"\[("[\w ]+", )*"[\w ]+"\]"#).unwrap())
            .then(LiteralParser::new(" },\n"))
            .repeat(4..=4)
            .then(LiteralParser::new("]"));
        let stream = llm.stream_structured_text(prompt, validator).await.unwrap();

        time_stream(stream).await;
    }

    {
        println!("\n\n# without constraints");
        print!("{}", prompt);

        let stream = llm.stream_text(prompt).with_max_length(100).await.unwrap();

        time_stream(stream).await;
    }
}

async fn time_stream(mut stream: impl TextStream + Unpin) {
    let start_time = std::time::Instant::now();
    let mut tokens = 0;
    while let Some(token) = stream.next().await {
        tokens += 1;
        print!("{token}");
        std::io::stdout().flush().unwrap();
    }
    let elapsed = start_time.elapsed();
    println!("\n\nGenerated {} tokens in {:?}", tokens, elapsed);
    println!(
        "Tokens per second: {:.2}",
        tokens as f64 / elapsed.as_secs_f64()
    );
}
