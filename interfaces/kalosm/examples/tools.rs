use kalosm::language::*;

#[tokio::main]
async fn main() {
    let llm = Phi::v2().unwrap();

    println!("Loading local documents...");
    let mut document_database = DocumentDatabase::new(
        Bert::builder().build().unwrap(),
        ChunkStrategy::Sentence {
            sentence_count: 1,
            overlap: 0,
        },
    );
    let documents = DocumentFolder::try_from(std::path::PathBuf::from("./documents")).unwrap();
    document_database.extend(documents).await.unwrap();
    println!("Loaded local documents.");

    let question = prompt_input("Question: ").unwrap();

    let mut tools = ToolManager::default().with_tool(DocumentSearchTool::new(document_database, 5));
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
