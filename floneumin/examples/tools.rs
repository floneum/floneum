use floneumin_language::floneumin_sample::CreateParserState;
use floneumin_language::floneumin_sample::Parser;
use floneumin_language::{floneumin_sample::StructuredSampler, *};
use futures_util::stream::StreamExt;
use llm_samplers::types::SamplerChain;
use rand::Rng;
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
                                    println!("\nResult: {:?}", result);
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

        let mut current_text = current_text.write().unwrap();
        let mut lines = current_text.lines().rev();
        let last_line = lines.next();
        if let Some(last_line) = last_line {
            if last_line.starts_with("Final Answer: ") {
                break;
            }
            if last_line.starts_with("Action Input: ") {
                let action_line = lines.next().unwrap();
                let tool = action_line.rsplit_once("Action: ").unwrap().1;
                let tool_input = last_line.rsplit_once("Action Input: ").unwrap().1;
                let mut tools = tools.write().unwrap();
                let tool = tools.get_tool_mut(tool).unwrap();
                let output = tool.run(tool_input).await;
                let observation = format!("Observation: {}", output);
                println!("{}", observation);
                current_text.push_str(&observation);

                current_text.push_str("\n");
            }
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
