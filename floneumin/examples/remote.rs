// You must set the environment variable OPENAI_API_KEY (https://platform.openai.com/account/api-keys) to run this example.

use std::io::Write;

use floneumin_language::*;
use floneumin_streams::TextStream;
use futures_util::stream::StreamExt;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let mut llm = Gpt4::start().await;
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    print!("{}", prompt);

    let stream = llm.stream_text(prompt).with_max_length(300).await.unwrap();

    let mut sentences = stream.words();
    while let Some(text) = sentences.next().await {
        print!("{}", text);
        std::io::stdout().flush().unwrap();
    }
}
