#![allow(unused)]
use html_parser::Element;
use kalosm::language::*;
use kalosm_llama::{EvaluationTrie, LlamaModel};
use std::{io::Write, sync::Arc};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();

    // let sampler = GenerationParameters::new().with_seed(0);
    let sampler = GenerationParameters::new();

    let mut llm = LlamaModel::from_builder(
        Llama::builder().with_source(LlamaSource::llama_3_2_3b_chat()),
        ModelLoadingProgress::multi_bar_loading_indicator(),
    )
    .await
    .unwrap();

    tokio::task::spawn_blocking(move || {
       let mut trie = EvaluationTrie::new();
        let mut last_entropy = 0.0;
        let task = prompt_input("Enter a task: ").unwrap();
        for generation in 0.. {
           let mut session = llm.new_session();
           let parser = Element::new_parser().repeat(1..=10).then(LiteralParser::new("<|im_end|>\n<|im_start|>user\n"));

           let output = llm.generate_structured_with_trie(
               &mut session,
               &format!("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{task}<|im_end|>\n<|im_start|>assistant\n"),
               sampler.clone(),
               parser,
               |token| {
                   print!("{}", token);
                   std::io::stdout().flush().unwrap();
                   Ok(())
               },
               &mut trie,
           ).unwrap();
           println!("\n\n");

            println!("generation {generation}:\n{output:?}");
            let mut shannon_entropy = trie.shannon_entropy();
            let entropy_diff = last_entropy - shannon_entropy;
            println!("entropy diff: {entropy_diff}");
            if entropy_diff.abs() < 0.00001 {
                println!("looks like entropy is converging, stopping generation");
                break;
            }
            println!("shannon entropy: {shannon_entropy}");
            last_entropy = shannon_entropy;
       }
   }).await.unwrap();
}
