use std::io::Write;

use floneumin_language::{
    local::LocalSession,
    model::{GenerationParameters, LlamaThirteenChatSpace, Model},
};
use floneumin_streams::text_stream::TextStream;
use futures_util::stream::StreamExt;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    let mut words = LocalSession::<LlamaThirteenChatSpace>::start()
        .await
        .stream_text(prompt, GenerationParameters::default().with_max_length(300))
        .await
        .unwrap()
        .words();

    print!("{}", prompt);

    while let Some(text) = words.next().await {
        print!("{}", text);
        std::io::stdout().flush().unwrap();
    }
}
