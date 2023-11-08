use kalosm_language::*;
use std::io::Write;

#[tokio::main]
async fn main() {
    let mut llm = Phi::start().await;

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

    print!("Question: ");
    std::io::stdout().flush().unwrap();
    let mut question = String::new();
    std::io::stdin().read_line(&mut question).unwrap();

    let mut tools = ToolManager::default().with_tool(DocumentSearchTool::new(document_database, 5));
    llm.run_sync(|llm| {
        Box::pin(async move {
            let prompt = tools.prompt(question);
            let mut session = llm.new_session().unwrap();
            llm.feed_text(&mut session, &prompt).unwrap();
            loop {
                if let ToolManagerStepResult::Finished(result) = tools
                    .run_step(llm, &mut session, |token| {
                        print!("{}", token);
                        std::io::stdout().flush().unwrap();
                        Ok(())
                    })
                    .await
                    .unwrap()
                {
                    println!();
                    println!("\n\nAnswer: {}", result);
                    break;
                }
            }
        })
    })
    .await
    .unwrap();
    std::future::pending::<()>().await;
}
