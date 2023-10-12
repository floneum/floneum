use floneumin_language::model::CreateModel;
use floneumin_language::tool::*;
use floneumin_language::{
    local::Phi,
    model::{GenerationParameters, Model},
};
use floneumin_streams::text_stream::TextStream;
use futures_util::stream::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() {
    let mut llm = Phi::start().await;
    let question = "What is the capital of France?";
    let tools = ToolManager::default().with_tool(WebSearchTool);
    let prompt = tools.prompt(question);
    let mut current_text = String::new();

    let stream = llm
        .stream_text(
            &prompt,
            GenerationParameters::default()
                .with_max_length(300)
                .with_stop_on("Action:".to_string()),
        )
        .await
        .unwrap();

    current_text.push_str(&prompt);
    print!("{}", prompt);

    let mut sentences = stream.words();
    while let Some(text) = sentences.next().await {
        print!("{}", text);
        current_text.push_str(&text);
        std::io::stdout().flush().unwrap();
    }
}
