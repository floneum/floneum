use futures_util::stream::StreamExt;
use kalosm_language::*;
use kalosm_sample::*;
use std::io::Write;
use std::sync::Arc;
use std::sync::Mutex;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let mut llm = Phi::start().await;
    let prompt = "Five US states";
    print!("{}", prompt);

    let validator = LiteralParser::from(": ")
        .then((StringParser::new(1..=20))
        .then(LiteralParser::from(", ")).repeat(1..=5));
    let validator_state = validator.create_parser_state();
    let mut words = llm
        .stream_structured_text_with_sampler(
            prompt,
            validator,
            validator_state,
            Arc::new(Mutex::new(
                GenerationParameters::default().bias_only_sampler(),
            )),
            Arc::new(Mutex::new(
                GenerationParameters::default().mirostat2_sampler(),
            )),
        )
        .await
        .unwrap();

    while let Some(text) = words.next().await {
        print!("{}", text);
        std::io::stdout().flush().unwrap();
    }
}
