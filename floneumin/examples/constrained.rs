use floneumin_language::*;
use floneumin_sample::*;
use futures_util::stream::StreamExt;
use llm_samplers::prelude::SamplerChain;
use std::io::Write;
use std::sync::Arc;
use std::sync::Mutex;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let mut llm = Phi::start().await;
    let prompt = "";
    print!("{}", prompt);

    let validator = StructureParser::Sequence {
        item: Box::new(StructureParser::Literal("Web Search".into())),
        separator: Box::new(StructureParser::Literal(", ".into())),
        min_len: 1,
        max_len: 2,
    };
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
