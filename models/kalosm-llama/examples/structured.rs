use cfg::{parse::Grammar, slab_grammar::SlabGrammar, tokenizer::Tokenizer, *};
use kalosm::language::*;
use kalosm_llama::{EvaluationTrie, LlamaModel};
use std::io::Write;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();

    // let sampler = GenerationParameters::new().with_seed(0);
    let sampler = GenerationParameters::new();

    let mut llm = LlamaModel::from_builder(
        Llama::builder().with_source(LlamaSource::llama_3_2_1b_chat()),
        ModelLoadingProgress::multi_bar_loading_indicator(),
    )
    .await
    .unwrap();

    let grammar = create_grammar();

    tokio::task::spawn_blocking(move || {
        let bump = bumpalo::Bump::new();
        let grammar = grammar.reallocate(&bump);
        let mut trie = EvaluationTrie::new();
        let mut last_entropy = 0.0;
        let task = include_str!("../../../sygus/src/prompt");
        for generation in 0.. {
            let mut session = llm.new_session();

            let output = llm.generate_structured_with_trie(
                &mut session,
                &format!("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{task}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
                sampler.clone(),
                Some(&grammar),
                todo!(),
                |token, _| {
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

fn create_grammar() -> Grammar<u32> {
    let tokenizer = Tokenizer::load_tokenizer("tokenizer.json");

    let grammar = parse::Grammar::parse(
        r#"Start -> '{' WHITESPACE '"settings"' WHITESPACE ':' WHITESPACE SETTINGS WHITESPACE '}'
        SETTINGS -> '{' WHITESPACE '"printInEndpoint"' WHITESPACE ':' WHITESPACE BOOLEAN WHITESPACE '}'
        BOOLEAN -> 'true' | 'false'
        WHITESPACE -> ε | ' ' | '   '"#,
    )
    .unwrap();

    let mut grammar = grammar.split_terminals();
    println!("Converting grammar to CNF...");
    grammar = grammar.to_cnf().unwrap();
    let grammar = grammar.replace_tokenizer_terminals(&tokenizer);
    let mut grammar = SlabGrammar::new(&grammar);
    grammar.garbage_collect_non_terminals();
    println!("start rule count: {}", grammar.rules.len());
    let merges = &tokenizer.merges;
    let mut processed_merges = Vec::new();
    for (i, merge) in merges.iter().enumerate() {
        println!(
            "Applying merge {i}: {:?} + {:?} = {:?}",
            String::from_utf8_lossy(&tokenizer.inverse_vocab[&merge.pair[0]]),
            String::from_utf8_lossy(&tokenizer.inverse_vocab[&merge.pair[1]]),
            String::from_utf8_lossy(&tokenizer.inverse_vocab[&merge.new_token])
        );
        grammar.shortcut_merge(merge, true);
        processed_merges.push(merge.clone());
    }

    let grammar = grammar.to_grammar();
    println!(
        "grammar:\n{}",
        grammar.clone().map(
            |r| String::from_utf8_lossy(&tokenizer.inverse_vocab[&r]).to_string(),
            |r| r.to_string()
        )
    );
    println!("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉");

    grammar
}
