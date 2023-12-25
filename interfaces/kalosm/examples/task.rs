use kalosm::language::*;

#[tokio::main]
async fn main() {
    let mut llm = Llama::new_chat();

    let mut task = Task::new(&mut llm, "You are a math assistant who helps students with their homework. You solve equations and answer questions. When solving problems, you will always solve problems step by step.");

    let start_timestamp = std::time::Instant::now();
    println!("question 1");
    // The first time we use the task, it will load the model and prompt.
    task.run("What is 2 + 2?")
        .await
        .unwrap()
        .to_std_out()
        .await
        .unwrap();
    println!("\nfirst question took: {:?}", start_timestamp.elapsed());

    let start_timestamp = std::time::Instant::now();
    println!("question 2");
    // After the first time, the model and prompt are cached.
    task.run("What is 4 + 4?")
        .await
        .unwrap()
        .to_std_out()
        .await
        .unwrap();
    println!("\nsecond question took: {:?}", start_timestamp.elapsed());

    let start_timestamp = std::time::Instant::now();
    println!("question 3");
    task.run("What is (7 + 4)*2?")
        .await
        .unwrap()
        .to_std_out()
        .await
        .unwrap();
    println!("\nthird question took: {:?}", start_timestamp.elapsed());
}
