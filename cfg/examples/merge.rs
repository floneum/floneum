use cfg::{
    parse::Grammar,
    slab_grammar::SlabGrammar,
    tokenizer::{Merge, Tokenizer},
    *,
};

fn main() {
    let log_every_n = std::env::var("LOG_EVERY_N")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    let verbose = std::env::var("VERBOSE").is_ok();
    let test = std::env::var("TEST").is_ok();
    let force_test = std::env::var("FORCE_TEST").is_ok();
    let without_cnf = std::env::var("WITHOUT_CNF").is_ok();
    let allow_incorrect = std::env::var("ALLOW_INCORRECT").is_ok();

    let tokenizer = Tokenizer::load_tokenizer("tokenizer.json");

    let grammar = parse::Grammar::parse(
        r#"Start -> ntString
ntString -> 'name' | '" "' | '(' 'str.++' ' ' ntString ' ' ntString ')' | '(' 'str.replace' ' ' ntString ' ' ntString ' ' ntString ')' | '(' 'str.at' ' ' ntString ' ' ntInt ')' | '(' 'int.to.str' ' ' ntInt ')' | '(' 'str.substr' ' ' ntString ' ' ntInt ' ' ntInt ')'
ntInt -> '0' | '1' | '2' | '(' '+' ' ' ntInt ' ' ntInt ')' | '(' '-' ' ' ntInt ' ' ntInt ')' | '(' 'str.len' ' ' ntString ')' | '(' 'str.to.int' ' ' ntString ')' | '(' 'str.indexof' ' ' ntString ' ' ntString ' ' ntInt ')'
ntBool -> 'true' | 'false' | '(' 'str.prefixof' ' ' ntString ' ' ntString ')' | '(' 'str.suffixof' ' ' ntString ' ' ntString ')' | '(' 'str.contains' ' ' ntString ' ' ntString ')'"#,
    )
    .unwrap();

    let mut grammar = grammar.split_terminals();
    if !without_cnf {
        println!("Converting grammar to CNF...");
        grammar = grammar.to_cnf().unwrap();
    }
    let bump = bumpalo::Bump::new();
    let grammar = grammar.replace_tokenizer_terminals(&tokenizer);
    let mut grammar = SlabGrammar::new(&grammar);
    grammar.garbage_collect_non_terminals();
    println!("start rule count: {}", grammar.rules.len());
    let merges = &tokenizer.merges;
    let mut last_size = grammar.rules.len();
    let mut processed_merges = Vec::new();
    for (i, merge) in merges.iter().enumerate() {
        println!(
            "Applying merge {i}: {:?} + {:?} = {:?}",
            String::from_utf8_lossy(&tokenizer.inverse_vocab[&merge.pair[0]]),
            String::from_utf8_lossy(&tokenizer.inverse_vocab[&merge.pair[1]]),
            String::from_utf8_lossy(&tokenizer.inverse_vocab[&merge.new_token])
        );
        if let Some(log_every_n) = log_every_n {
            if i % log_every_n == 0 {
                let grammar = grammar.to_grammar();
                println!(
                    "grammar before:\n{}",
                    grammar.clone().map(
                        |r| String::from_utf8_lossy(&tokenizer.inverse_vocab[&r]).to_string(),
                        |r| r.to_string()
                    )
                );
            }
        }
        let start = std::time::Instant::now();
        let changed = grammar.shortcut_merge(merge, allow_incorrect);
        processed_merges.push(merge.clone());
        if changed {
            println!("Time to merge: {:?}", start.elapsed());
            println!("size: {}", grammar.rules.len());
        }
        if verbose {
            let grammar = grammar.to_grammar();
            println!(
                "grammar before gc:\n{}",
                grammar.clone().map(
                    |r| String::from_utf8_lossy(&tokenizer.inverse_vocab[&r]).to_string(),
                    |r| r.to_string()
                )
            );
        }
        if changed {
            let start = std::time::Instant::now();
            grammar.inline_optimize();
            println!("Time to garbage collect: {:?}", start.elapsed());
            println!("after merge rule count: {}", grammar.rules.len());
            println!(
                "grew by a factor of {:.10}",
                grammar.rules.len() as f64 / last_size as f64
            );
        }
        if let Some(log_every_n) = log_every_n {
            if i % log_every_n == 0 {
                let grammar = grammar.to_grammar();
                println!(
                    "grammar after:\n{}",
                    grammar.clone().map(
                        |r| String::from_utf8_lossy(&tokenizer.inverse_vocab[&r]).to_string(),
                        |r| r.to_string()
                    )
                );
            }
        }
        last_size = grammar.rules.len();
        if (test && changed) || force_test {
            let grammar = grammar.to_grammar();
            if verbose {
                println!(
                    "grammar:\n{}",
                    grammar.clone().map(
                        |r| String::from_utf8_lossy(&tokenizer.inverse_vocab[&r]).to_string(),
                        |r| r.to_string()
                    )
                );
            }
            run_test(
                grammar,
                &tokenizer,
                &processed_merges,
                &bump,
                allow_incorrect,
            );
        }
        grammar.verify_integrity("after shortcut merge");
    }

    // grammar.inline_optimize();
    let grammar = grammar.to_grammar();
    println!(
        "grammar:\n{}",
        grammar.clone().map(
            |r| String::from_utf8_lossy(&tokenizer.inverse_vocab[&r]).to_string(),
            |r| r.to_string()
        )
    );
    println!("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉");
    run_test(
        grammar,
        &tokenizer,
        &processed_merges,
        &bump,
        allow_incorrect,
    );
}

fn run_test(
    grammar: Grammar<u32>,
    tokenizer: &Tokenizer,
    processed_merges: &[Merge],
    bump: &bumpalo::Bump,
    allow_incorrect: bool,
) {
    let test_time = std::time::Instant::now();
    let dense_grammar = grammar.reallocate(&bump);
    println!("dense size: {}", bump.allocated_bytes());
    println!("after shortcut merge rule count: {}", grammar.rules.len());

    let should_recognize: &[&'static [u8]] = &[
        br#"name"#,
        br#"" ""#,
        br#"(str.++ name " ")"#,
        br#"(str.++ (str.++ name " ") name)"#,
    ];
    for input in should_recognize {
        let mut passing = true;
        let mut bytes = Vec::new();
        for byte in *input {
            bytes.push(tokenizer.bytes[*byte as usize]);
        }
        let mut tokenized = bytes.clone();
        for merge in processed_merges {
            let mut new = tokenized.clone();
            let mut i = 0;
            while i < new.len() - 1 {
                // Replace the pair of tokens with the new token if they match the merge
                if new[i] == merge.pair[0] && new.get(i + 1) == Some(&merge.pair[1]) {
                    new[i] = merge.new_token;
                    new.remove(i + 1);
                } else {
                    i += 1;
                }
            }
            if new != tokenized {
                // If the new tokenized version is different, we have a merge
                // Make sure the incorrectly tokenized version is not recognized
                let recognizes = dense_grammar.recognizes_tokens(tokenized.iter().copied());
                if recognizes != allow_incorrect {
                    eprintln!(
                        "recognized incorrectly tokenized version {:?}",
                        String::from_utf8_lossy(input)
                    );
                    passing = false;
                }
            }
            tokenized = new;
        }
        assert!(
            dense_grammar.recognizes_tokens(tokenized.iter().copied()),
            "Failed to recognize input: {:?} after tokenizing into {:?}",
            input,
            tokenized
                .iter()
                .map(|b| String::from_utf8_lossy(&tokenizer.inverse_vocab[b]).to_string())
                .collect::<Vec<_>>()
        );
        if !passing {
            panic!();
        }
    }
    println!("Time to test: {:?}", test_time.elapsed());
}
