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
    let prompt = "A";
    print!("{}", prompt);

    let validator = LiteralParser::from(" list of 10 numbers: ")
        .then(IntegerParser::new(1004..=1005))
        .then(LiteralParser::from(", "))
        .then(IntegerParser::new(-1005..=-1000))
        .then(LiteralParser::from(", "))
        .then(IntegerParser::new(0..=10))
        .then(LiteralParser::from(", "))
        .then(IntegerParser::new(0..=10))
        .then(LiteralParser::from(", "))
        .then(IntegerParser::new(0..=10))
        .then(LiteralParser::from(", "))
        .then(IntegerParser::new(0..=10))
        .then(LiteralParser::from(", "))
        .then(IntegerParser::new(0..=10))
        .then(LiteralParser::from(", "))
        .then(IntegerParser::new(0..=10))
        .then(LiteralParser::from(", "))
        .then(IntegerParser::new(0..=10))
        .then(LiteralParser::from(", "))
        .then(IntegerParser::new(0..=10));
    let validator_state = validator.create_parser_state();
    let mut words = llm
        .stream_structured_text_with_sampler(
            prompt,
            validator,
            validator_state,
            Arc::new(Mutex::new(GenerationParameters::default().bias_only_sampler())),
        )
        .await
        .unwrap();

    while let Some(text) = words.next().await {
        print!("{}", text);
        std::io::stdout().flush().unwrap();
    }
}
