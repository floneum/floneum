use kalosm::language::*;

#[tokio::main]
async fn main() {
    let llm = Llama::default();

    let question = prompt_input("Question: ").unwrap();

    let mut tools = ToolManager::default().with_tool(CalculatorTool);
    llm.run_sync(|llm| {
        Box::pin(async move {
            let mut prompt = tools.prompt(question);
            let mut session = llm.new_session().unwrap();
            loop {
                match tools
                    .run_step(&prompt, llm, &mut session, |_| Ok(()))
                    .await
                    .unwrap()
                {
                    ToolManagerStepResult::Finished(result) => {
                        println!("\n\nAnswer: {}", result);
                        break;
                    }
                    ToolManagerStepResult::Action(result) => {
                        prompt = format!("{result}\n");
                        println!("Action Result: {}", result)
                    }
                    ToolManagerStepResult::Thought(thought) => {
                        prompt = format!("{thought}\n");
                        println!("Thought: {}", thought);
                    }
                }
            }
        })
    })
    .unwrap();
    std::future::pending::<()>().await;
}
