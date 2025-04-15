#![allow(unused)]
use kalosm::language::*;
use kalosm_llama::{EvaluationTrie, LlamaModel};
use std::{io::Write, sync::Arc};

#[tokio::main]
async fn main() {

    #[derive(Debug, Clone, Parse)]
    struct Pet {
        #[parse(len = 1..=10)]
        name: String,
        #[parse(len = 1..=20)]
        description: String,
        #[parse(len = 1..=10)]
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

   // let sampler = GenerationParameters::new().with_seed(0);
   let sampler = GenerationParameters::new();
    
   let mut llm = LlamaModel::from_builder(
       Llama::builder().with_source(LlamaSource::qwen_2_5_0_5b_instruct()),
       ModelLoadingProgress::multi_bar_loading_indicator(),
   )
   .await
   .unwrap();

   tokio::task::spawn_blocking(move || {
       let mut trie = EvaluationTrie::new();
       for generation in 0.. {
           let mut session = llm.new_session();

           let output = llm.generate_structured_with_trie(
               &mut session,
               "Generate a JSON object with the following properties: name, description, color, size, diet",
               sampler.clone(),
               Pet::new_parser(),
               |token| {
                   // print!("{}", token);
                   // std::io::stdout().flush().unwrap();
                   Ok(())
               },
               &mut trie,
           ).unwrap();

           println!("generation {generation}:\n{output:?}");
       }
   }).await.unwrap();
}