//! This example works for any endpoint that has the same interface as OpenAI's API.
//! This can be useful if you want to self-host a remote model.
//!
//! If you would like to self host a llama model, you can use a tool like litellm to host the model: https://github.com/BerriAI/litellm#openai-proxy---docs

use kalosm::language::*;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let base_url = std::env::var("OPENAI_API_BASE").expect("Custom OPENAI_API_BASE not set");
    let llm = Gpt4::builder().with_base_url(&base_url).build();
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    print!("{}", prompt);

    let stream = llm.stream_text(prompt).with_max_length(300).await.unwrap();
    stream.to_std_out().await.unwrap();
}
