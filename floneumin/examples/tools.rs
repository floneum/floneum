use floneumin_language::{floneumin_sample::StructuredSampler, *};
use futures_util::stream::StreamExt;
use llm_samplers::types::SamplerChain;
use std::{
    io::Write,
    sync::{Arc, Mutex},
};

#[tokio::main]
async fn main() {
    let question = "Find the average of 5, 6, 7, and 123";
    let mut tools = ToolManager::default()
        .with_tool(WebSearchTool::new(1))
        .with_tool(CalculatorTool);

    let mut current_text = tools.prompt(question);
    print!("{}", current_text);
    for _ in 0..4 {
        // TODO: There seems to be a bug in candle that causes reusing the session to fail here
        let mut llm = Phi::start().await;
        let validator = tools.any_action_constraint();
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
                let tool = tools.get_tool_mut(tool).unwrap();
                let output = tool.run(tool_input).await;
                let observation = format!("Observation: {}", output);
                println!("{}", observation);
                current_text.push_str(&observation);

                current_text.push_str("\n");
            }
        }
    }
    // TODO: There seems to be a bug in candle that causes reusing the session to fail here
    let mut llm = Phi::start().await;
    let validator = tools.answer_constraints();
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
