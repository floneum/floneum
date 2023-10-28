use futures_util::stream::StreamExt;
use kalosm_language::*;
use kalosm_sample::*;
use llm_samplers::prelude::SamplerChain;
use std::io::Write;
use std::sync::Arc;
use std::sync::Mutex;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let mut llm = Phi::start().await;
    let prompt = "A";
    print!("{}", prompt);

    let validator = LiteralParser::from("A list of 10 numbers: ")
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
    let structured = StructuredSampler::new(validator.clone(), 0, llm.tokenizer());
    let chain = SamplerChain::new() + structured;
    let mut words = llm
        .stream_text_with_sampler(prompt, Some(300), None, Arc::new(Mutex::new(chain)))
        .await
        .unwrap();

    while let Some(text) = words.next().await {
        print!("{}", text);
        std::io::stdout().flush().unwrap();
    }
}
