use floneumin_language::{
    floneumin_sample::{CreateParserState, LiteralParser, Parser},
    *,
};
use std::io::Write;

#[tokio::main]
async fn main() {
    let mut llm = Phi::start().await;
    let prompt = "this";
    let parser = LiteralParser::from("this is a test of how quickly this can be done.".repeat(100));
    print!("{}", prompt);

    let tokenizer = llm.tokenizer();
    llm.run_sync(Box::new(move |llm| {
        let mut current_text = prompt.to_string();
        let mut parser_state = parser.create_parser_state();
        match parser.parse(&parser_state, prompt.as_bytes()).unwrap() {
            floneumin_sample::ParseResult::Incomplete(new_state) => {
                parser_state = new_state;
            }
            floneumin_sample::ParseResult::Finished { result, .. } => {
                println!("Result: {:?}", result);
                return;
            }
        }
        loop {
            let mut logits = llm.run(&current_text).unwrap();
            logits.sort_by(|a, b| a.prob.partial_cmp(&b.prob).unwrap());
            let mut best_token = None;
            for logit in logits.iter() {
                let new_text = tokenizer.decode(&[logit.token_id]).unwrap();
                if new_text.is_empty() {
                    continue;
                }
                if let Ok(result) = parser.parse(&parser_state, new_text.as_bytes()) {
                    match result {
                        floneumin_sample::ParseResult::Incomplete(new_state) => {
                            best_token = Some((new_text.to_string(), new_state));
                            break;
                        }
                        floneumin_sample::ParseResult::Finished { result, .. } => {
                            print!("{}", new_text);
                            std::io::stdout().flush().unwrap();
                            println!("\nResult: {:?}", result);
                            return;
                        }
                    }
                }
            }
            if let Some((token, new_state)) = best_token {
                print!("{}", token);
                std::io::stdout().flush().unwrap();
                current_text.push_str(&token);
                parser_state = new_state;
            } else {
                println!("No valid tokens found");
                break;
            }
        }
    }))
    .await
    .unwrap();
}
