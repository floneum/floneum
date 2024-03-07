use kalosm::language::*;

#[tokio::main]
async fn main() {
    let llm = Llama::new().await.unwrap();
    let prompt = "Five prime numbers: 2, ";

    println!("# with constraints");
    print!("{}", prompt);

    let validator = RegexParser::new(r"(\d, ){4}\d").unwrap();
    let stream = llm.stream_structured_text(prompt, validator).await.unwrap();

    stream.split().0.to_std_out().await.unwrap();

    println!("\n\n# without constraints");
    print!("{}", prompt);

    let stream = llm.stream_text(prompt).with_max_length(100).await.unwrap();
    stream.to_std_out().await.unwrap();
}
