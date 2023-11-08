use kalosm_language::*;
use kalosm_sample::*;
use std::sync::Arc;
use std::sync::Mutex;

#[tokio::main]
async fn main() {
    let mut llm = Phi::start().await;
    let prompt = "Realistic mock user names for a chat application: ";

    let validator = <[Word<1, 10>; 10] as HasParser>::new_parser();
    let validator_state = validator.create_parser_state();
    let words = llm
        .stream_structured_text_with_sampler(
            prompt,
            validator,
            validator_state,
            Arc::new(Mutex::new(GenerationParameters::default().sampler())),
        )
        .await
        .unwrap();

    println!("\n{:#?}", words.result().await);
}
