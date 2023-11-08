use kalosm_language::kalosm_sample::CreateParserState;
use kalosm_language::kalosm_sample::Either;
use kalosm_language::*;
use std::io::Write;
use std::sync::Arc;
use std::sync::Mutex;

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
    let mut tools = ToolManager::default().with_tool(DocumentSearchTool::new(document_database, 5));

    loop {
        print!("Question: ");
        std::io::stdout().flush().unwrap();
        let mut question = String::new();
        std::io::stdin().read_line(&mut question).unwrap();

        let mut current_text = tools.prompt(&question);

        loop {
            let constraints = tools.any_action_constraint();
            let validator_state = constraints.create_parser_state();
            let mut words = llm
                .stream_structured_text_with_sampler(
                    &current_text,
                    constraints,
                    validator_state,
                    Arc::new(Mutex::new(GenerationParameters::default().sampler())),
                )
                .await
                .unwrap();

            while let Some(text) = words.next().await {
                print!("{}", text);
                current_text += &text;
                std::io::stdout().flush().unwrap();
            }

            match words.result().await.unwrap() {
                Either::Left(Either::Left(_)) => {}
                Either::Left(Either::Right(((), (tool_index, ((), left))))) => {
                    let result = tools
                        .get_tool_mut_by_index(tool_index)
                        .unwrap()
                        .run(left)
                        .await;
                    current_text += "\n";
                    current_text += &result;
                    current_text += "\nRemember: You are answering the question: ";
                    current_text += &question;
                    current_text += "\n";
                    println!("Tool Result: {}", result);
                }
                Either::Right(right) => {
                    println!("Final Answer: {}", right.1);
                    break;
                }
            }
        }
    }
}
