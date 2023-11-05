use futures_util::stream::StreamExt;
use kalosm_language::*;
use kalosm_sample::*;
use std::io::Write;
use std::sync::Arc;
use std::sync::Mutex;

#[tokio::main]
async fn main() {
    let mut llm = Phi::start().await;
    let prompt = "An array of numbers from 1 to 10: ";

    println!("# with constraints");
    print!("{}", prompt);
    let validator = <[u32; 10] as HasParser>::new_parser();
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

    println!("\n{:#?}", words.result().await);

    println!("\n\n# without constraints");
    print!("{}", prompt);

    let mut words = llm.stream_text(prompt).with_max_length(100).await.unwrap();
    while let Some(text) = words.next().await {
        print!("{}", text);
        std::io::stdout().flush().unwrap();
    }
}
