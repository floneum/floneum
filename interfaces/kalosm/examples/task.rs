use kalosm::language::*;

#[tokio::main]
async fn main() {
    let mut llm = Phi::v2();

    let operator = IndexParser::new(vec![
        LiteralParser::new("+"),
        LiteralParser::new("-"),
        LiteralParser::new("*"),
        LiteralParser::new("/"),
    ]);
    let equation = IntegerParser::new(0..=100)
        .then(operator)
        .then(IntegerParser::new(0..=100))
        .then(LiteralParser::new(" = "))
        .then(IntegerParser::new(0..=100));
    let calculation_step = LiteralParser::new("Step ")
        .then(IntegerParser::new(1..=9))
        .then(LiteralParser::new(": "))
        .then(equation)
        .then(LiteralParser::new("\n"));
    let output = LiteralParser::new("Output: ").then(IntegerParser::new(0..=100));

    let task = Task::builder(&llm, "You are an assistant who solves math problems. When solving problems, you will always solve problems step by step with one step per line.")
        .with_constraints(calculation_step.repeat(0..=2).then(output))
        .with_example("What is 1 + 2?", "Step 1: 1+2 = 3\nOutput: 3")
        .with_example("What is 3 + 4?", "Step 1: 3+4 = 7\nOutput: 7")
        .with_example("What is (1 + 2) * 3?", "Step 1: 1+2 = 3\nStep 2: 3*3 = 9\nOutput: 9")
        .build();

    let start_timestamp = std::time::Instant::now();
    println!("question 1");
    // The first time we use the task, it will load the model and prompt.
    task.run("What is 2 + 2?", &mut llm)
        .await
        .unwrap()
        .split()
        .0
        .to_std_out()
        .await
        .unwrap();
    println!("\nfirst question took: {:?}", start_timestamp.elapsed());

    let start_timestamp = std::time::Instant::now();
    println!("question 2");
    // After the first time, the model and prompt are cached.
    task.run("What is 4 + 4?", &mut llm)
        .await
        .unwrap()
        .split()
        .0
        .to_std_out()
        .await
        .unwrap();
    println!("\nsecond question took: {:?}", start_timestamp.elapsed());

    let start_timestamp = std::time::Instant::now();
    println!("question 3");
    task.run("What is (7 + 4)*2?", &mut llm)
        .await
        .unwrap()
        .split()
        .0
        .to_std_out()
        .await
        .unwrap();
    println!("\nthird question took: {:?}", start_timestamp.elapsed());
}
