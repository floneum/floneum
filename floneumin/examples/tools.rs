use floneumin_language::floneumin_sample::CreateParserState;
use floneumin_language::floneumin_sample::Parser;
use floneumin_language::*;
use rand::Rng;
use std::io::Write;

const TOP_K: usize = 10;
const TOP_P: f32 = 10.0;

#[tokio::main]
async fn main() {
    let mut llm = Phi::start().await;

    let tokenizer = llm.tokenizer();
    llm.run_sync(Box::new({
        move |llm| {
            Box::pin(async move {
                let question = "What is 10 + 10?";
                let mut tools = ToolManager::default()
                    // .with_tool(WebSearchTool::new(1))
                    .with_tool(CalculatorTool);

                let mut current_text = tools.prompt(question);
                let mut bytes_fed = 0;
                print!("{}", current_text);
                std::io::stdout().flush().unwrap();
                let parser = tools.any_action_constraint();

                for _ in 0..5 {
                    let mut parser_state = parser.create_parser_state();
                    loop {
                        let mut logits = llm.feed_text(current_text.split_at(bytes_fed).1).unwrap();
                        bytes_fed =current_text.len();
                        logits.ensure_sorted().unwrap();
                        let min_prob = logits.last().unwrap().logit;
                        for logit in logits.iter_mut() {
                            logit.logit = logit.logit - min_prob;
                            debug_assert!(logit.logit >= 0.0)
                        }
                        let mut choices = Vec::with_capacity(TOP_K);
                        let mut total_prob = 0.0;
                        for logit in logits.iter() {
                            let new_text = tokenizer.decode(&[logit.token_id]).unwrap();
                            if new_text.is_empty() || logit.logit == 0.0{
                                continue;
                            }
                            if let Ok(result) = parser.parse(&parser_state, new_text.as_bytes()) {
                                let result = result.without_remaining();
                                let prob = logit.logit;
                                total_prob += prob;
                                choices.push((new_text.to_string(), result, prob));
                                if choices.len() >= TOP_K && total_prob >= TOP_P {
                                    break;
                                }
                            }
                        }
                        if choices.is_empty() {
                            println!("No choices found");
                            break;
                        }
                        let total = choices.iter().map(|(_, _, prob)| prob).sum::<f32>();
                        let mut rng = rand::thread_rng();
                        let random_choice = if total == 0.0 {
                            0.0
                        } else {
                            rng.gen_range(0.0..total)
                        };
                        let mut best_token = None;

                        let mut total = 0.0;
                        for (token, result, prob) in choices {
                            total += prob;
                            if total >= random_choice {
                                best_token = Some((token, result));
                                break;
                            }
                        }
                        let (token, result) = best_token.unwrap();
                        current_text.push_str(&token);
                        print!("{}", token);
                        std::io::stdout().flush().unwrap();
                        match result {
                            floneumin_sample::ParseResult::Incomplete(new_state) => {
                                parser_state = new_state;
                            }
                            floneumin_sample::ParseResult::Finished { result, .. } => {
                                match result {
                                    floneumin_sample::Either::Right(((), result)) => {
                                        println!("\n\n\n\n\nfinal result is {result}");
                                        return;
                                    }
                                    floneumin_sample::Either::Left(
                                        floneumin_sample::Either::Right((((), tool_index), tool_input)),
                                    ) => {
                                        let tool = tools.get_tool_mut_by_index(tool_index).unwrap();
                                        let output = tool.run(&tool_input).await;
                                        let observation = format!("Observation: {}", output);
                                        println!("{}", observation);
                                        current_text.push_str(&observation);
                                        current_text.push_str("\n");
                                        
                                    }
                                    _=> {}
                                }
                                break;
                            }
                        }
                    }
                }
            })
        }
    }))
    .await
    .unwrap();
}
