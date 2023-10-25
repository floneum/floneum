use std::io::Write;

use kalosm_language::*;
use kalosm_streams::TextStream;
use futures_util::stream::StreamExt;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    let mut words = LocalSession::<LlamaThirteenChatSpace>::start()
        .await
        .stream_text(prompt)
        .with_max_length(300)
        .await
        .unwrap()
        .words();

    print!("{}", prompt);

    while let Some(text) = words.next().await {
        print!("{}", text);
        std::io::stdout().flush().unwrap();
    }
}
