use floneumin_language::model::GenerationParameters;
use floneumin_language::sample::structured::StructuredSampler;
use floneumin_language::sample::structured_parser::StructureParser;
use floneumin_language::{local::Phi, model::Model};
use floneumin_streams::text_stream::TextStream;
use futures_util::stream::StreamExt;
use llm_samplers::prelude::SamplerChain;
use std::io::Write;
use std::sync::Arc;
use std::sync::Mutex;

#[tokio::main]
async fn main() {
    let mut llm = Phi::start().await;
    let prompt = "\"";
    print!("{}", prompt);

    let structured = StructuredSampler::new(
        StructureParser::String {
            min_len: 1,
            max_len: 10,
        },
        0,
        llm.tokenizer(),
    );
    let chain = SamplerChain::new() + structured + GenerationParameters::default().sampler();
    let mut words = llm
        .stream_text_with_sampler(prompt, Some(300), Arc::new(Mutex::new(chain)))
        .await
        .unwrap()
        .words();

    while let Some(text) = words.next().await {
        print!("{}", text);
        std::io::stdout().flush().unwrap();
    }
}
