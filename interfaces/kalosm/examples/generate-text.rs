use kalosm::language::*;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    let llm = Llama::new().await.unwrap();

    print!("{prompt}");

    llm(prompt).to_std_out().await.unwrap();
}
