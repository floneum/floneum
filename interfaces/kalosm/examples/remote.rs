// You must set the environment variable OPENAI_API_KEY (https://platform.openai.com/account/api-keys) to run this example.

use kalosm::language::*;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let llm = Gpt3_5::default();
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    print!("{}", prompt);

    let mut stream = llm.stream_text(prompt).with_max_length(300).await.unwrap();
    stream.to_std_out().await.unwrap();
}
