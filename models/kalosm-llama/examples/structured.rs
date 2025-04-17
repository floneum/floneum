#![allow(unused)]
use kalosm::language::*;
use kalosm_llama::{EvaluationTrie, LlamaModel};
use std::{io::Write, sync::Arc};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();

    #[derive(Debug, Clone, Parse)]
    struct Pet {
        #[parse(len = 10..=20)]
        name: String,
        #[parse(len = 1..=40)]
        description: String,
        #[parse(len = 5..=10)]
        color: String,
        size: Size,
        diet: Diet,
    }

    #[derive(Debug, Clone, Parse)]
    enum Diet {
        #[parse(rename = "carnivore")]
        Carnivore,
        #[parse(rename = "herbivore")]
        Herbivore,
        #[parse(rename = "omnivore")]
        Omnivore,
    }

    #[derive(Debug, Clone, Parse)]
    enum Size {
        #[parse(rename = "small")]
        Small,
        #[parse(rename = "medium")]
        Medium,
        #[parse(rename = "large")]
        Large,
    }

    println!("# with constraints");

    // let sampler = GenerationParameters::new().with_seed(0);
    let sampler = GenerationParameters::new();

    let mut llm = LlamaModel::from_builder(
        Llama::builder().with_source(LlamaSource::llama_3_2_1b_chat()),
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

            let output = llm.generate_structured_with_trie(
                &mut session,
                &format!("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{task}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
                sampler.clone(),
                Pet::new_parser(),
                |token| {
                    print!("{}", token);
                    std::io::stdout().flush().unwrap();
                    Ok(())
                },
                &mut trie,
            ).unwrap();

            println!("generation {generation}:\n{output:?}");
            let shannon_entropy = trie.shannon_entropy();
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
