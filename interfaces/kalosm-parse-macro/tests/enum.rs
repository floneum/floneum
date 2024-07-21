#![allow(unused)]

use kalosm::language::*;

#[derive(Parse, Clone)]
#[parse(tag = "ty", content = "contents")]
enum NamedEnum {
    #[parse(rename = "person")]
    Person { name: String, age: u32 },
    #[parse(rename = "animal")]
    Animal { name: String, species: String },
}

#[tokio::test]
async fn named_enum() {
    let model = Llama::builder()
        .with_source(LlamaSource::tiny_llama_1_1b_chat())
        .build()
        .await
        .unwrap();

    let task = Task::builder("You generate json")
        .with_constraints(NamedEnum::new_parser())
        .build();

    let output = task
        .run("What is the capital of France?", &model)
        .all_text()
        .await;
    println!("{output}");

    assert!(output.contains("\"ty\":"));
    assert!(output.contains("\"contents\":"));
    assert!(output.contains("\"ty\": \"person\"") || output.contains("\"ty\": \"animal\""));
    assert!(output.contains("\"name\":"));
    assert!(output.contains("\"age\":") || output.contains("\"species\":"));
}

#[derive(Parse, Clone)]
enum MixedEnum {
    Person {
        #[parse(rename = "person")]
        name: String,
        age: u32,
    },
    Animal,
    Turtle(String),
}

#[tokio::test]
async fn mixed_enum() {
    let model = Llama::builder()
        .with_source(LlamaSource::tiny_llama_1_1b_chat())
        .build()
        .await
        .unwrap();

    let task = Task::builder("You generate json")
        .with_constraints(MixedEnum::new_parser())
        .build();

    let output = task
        .run("What is the capital of France?", &model)
        .all_text()
        .await;
    println!("{output}");
}

#[derive(Parse, Clone)]
enum UnitEnum {
    First,
    #[parse(rename = "second")]
    Second,
}

#[tokio::test]
async fn unit_enum() {
    let model = Llama::builder()
        .with_source(LlamaSource::tiny_llama_1_1b_chat())
        .build()
        .await
        .unwrap();

    let task = Task::builder("You generate json")
        .with_constraints(UnitEnum::new_parser())
        .build();

    let output = task
        .run("What is the capital of France?", &model)
        .all_text()
        .await;
    println!("{output}");
}

#[derive(Parse, Clone)]
enum TupleEnum {
    First(String),
    Second(String),
}

#[tokio::test]
async fn tuple_enum() {
    let model = Llama::builder()
        .with_source(LlamaSource::tiny_llama_1_1b_chat())
        .build()
        .await
        .unwrap();

    let task = Task::builder("You generate json")
        .with_constraints(TupleEnum::new_parser())
        .build();

    let output = task
        .run("What is the capital of France?", &model)
        .all_text()
        .await;
    println!("{output}");

    assert!(output.contains("\"type\": \"First\"") || output.contains("\"type\": \"Second\""));
}

#[test]
fn unit_enum_parses() {
    #[derive(Parse, Debug, Clone, PartialEq)]
    enum Color {
        Red,
        Blue,
        Green,
    }

    let parser = Color::new_parser();
    let state = parser.create_parser_state();
    let color = parser.parse(&state, b"\"Red\" ").unwrap().unwrap_finished();
    assert_eq!(color, Color::Red);
}
