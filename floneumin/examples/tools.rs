use floneumin_language::floneumin_sample::CreateParserState;
use floneumin_language::floneumin_sample::Parser;
use floneumin_language::{floneumin_sample::StructuredSampler, *};
use futures_util::stream::StreamExt;
use llm_samplers::types::SamplerChain;
use rand::Rng;
use std::cell::OnceCell;
use std::sync::atomic::AtomicUsize;
use std::{
    io::Write,
    sync::{Arc, Mutex, RwLock},
};

const TOP_K: usize = 10;
const TOP_P: f32 = 0.9;

#[tokio::main]
async fn main() {
    let question = "Find the average of 5, 6, 7, and 123";
    let tools = Arc::new(RwLock::new(
        ToolManager::default()
            .with_tool(WebSearchTool::new(1))
            .with_tool(CalculatorTool),
    ));

    let current_text = Arc::new(RwLock::new(tools.read().unwrap().prompt(question)));
    print!("{}", current_text.read().unwrap());
    let current_tool_index = Arc::new(AtomicUsize::new(0));
    let current_tool_input = Arc::new(Mutex::new(Some(String::new())));
    for _ in 0..4 {
        // TODO: There seems to be a bug in candle that causes reusing the session to fail here
        let mut llm = Phi::start().await;
        let parser = {
            let tools = tools.read().unwrap();
            tools.any_action_constraint()
        };

        let tokenizer = llm.tokenizer();
        llm.run_sync(Box::new({
            let current_text = current_text.clone();
            let current_tool_index = current_tool_index.clone();
            let current_tool_input = current_tool_input.clone();
            move |llm| {
                let mut parser_state = parser.create_parser_state();
                loop {
                    let mut logits = llm.run(&current_text.read().unwrap()).unwrap();
                    logits.sort_by(|a, b| b.logit.partial_cmp(&a.logit).unwrap());
                    let mut choices = Vec::with_capacity(TOP_K);
                    let mut total_prob = 0.0;
                    for logit in logits.iter() {
                        let new_text = tokenizer.decode(&[logit.token_id]).unwrap();
                        if new_text.is_empty() {
                            continue;
                        }
                        if let Ok(result) = parser.parse(&parser_state, new_text.as_bytes()) {
                            match result {
                                floneumin_sample::ParseResult::Incomplete(new_state) => {
                                    choices.push((new_text.to_string(), new_state, logit.logit));
                                    total_prob += logit.logit;
                                    if choices.len() >= TOP_K && total_prob >= TOP_P {
                                        break;
                                    }
                                    break;
                                }
                                floneumin_sample::ParseResult::Finished { result, .. } => {
                                    print!("{}", new_text);
                                    std::io::stdout().flush().unwrap();
                                    if let floneumin_sample::Either::Left(
                                        floneumin_sample::Either::Right((
                                            (((), tool_index), ()),
                                            input,
                                        )),
                                    ) = result
                                    {
                                        current_tool_index.store(
                                            tool_index,
                                            std::sync::atomic::Ordering::Relaxed,
                                        );
                                        *current_tool_input.lock().unwrap() = Some(input);
                                    }

                                    return;
                                }
                            }
                        }
                    }
                    if choices.is_empty() {
                        println!("No valid tokens found");
                        break;
                    }
                    let total = choices.iter().map(|(_, _, prob)| prob).sum::<f32>();
                    let mut rng = rand::thread_rng();
                    let random_choice = rng.gen_range(0.0..total);
                    let mut best_token = None;

                    let mut total = 0.0;
                    for (token, new_state, prob) in choices {
                        total += prob;
                        if total >= random_choice {
                            best_token = Some((token, new_state));
                            break;
                        }
                    }
                    let (token, new_state) = best_token.unwrap();
                    print!("{}", token);
                    std::io::stdout().flush().unwrap();
                    current_text.write().unwrap().push_str(&token);
                    parser_state = new_state;
                }
            }
        }))
        .await
        .unwrap();

        let tool_input = {
            let mut tool_input = current_tool_input.lock().unwrap();
            tool_input.take()
        };
        if let Some(tool_input) = tool_input {
            let tool = current_tool_index.load(std::sync::atomic::Ordering::Relaxed);
            let mut tools = tools.write().unwrap();
            let tool = tools.get_tool_mut_by_index(tool).unwrap();
            let output = tool.run(&tool_input).await;
            let observation = format!("Observation: {}", output);
            println!("{}", observation);
            let mut current_text = current_text.write().unwrap();
            current_text.push_str(&observation);

            current_text.push_str("\n");
        }
    }
    let mut current_text = current_text.write().unwrap();
    // TODO: There seems to be a bug in candle that causes reusing the session to fail here
    let mut llm = Phi::start().await;
    let validator = tools.read().unwrap().answer_constraints();
    let token_count = llm.tokenizer().encode(&current_text).unwrap().len();
    let structured = StructuredSampler::new(validator, token_count, llm.tokenizer());
    let chain = SamplerChain::new() + structured;
    let mut words = llm
        .stream_text_with_sampler(&current_text, Some(300), None, Arc::new(Mutex::new(chain)))
        .await
        .unwrap();

    while let Some(text) = words.next().await {
        print!("{}", text);
        current_text.push_str(&text);
        std::io::stdout().flush().unwrap();
    }
}
