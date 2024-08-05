#![allow(unused)]

use kalosm::language::*;

#[derive(Parse, Schema, Clone, PartialEq, Debug)]
#[parse(rename = "empty struct")]
struct EmptyNamedStruct {}

#[tokio::test]
async fn empty_struct() {
    let model = Llama::builder()
        .with_source(LlamaSource::tiny_llama_1_1b_chat())
        .build()
        .await
        .unwrap();

    let task = Task::builder("You generate json")
        .with_constraints(EmptyNamedStruct::new_parser())
        .build();

    let output = task
        .run("What is the capital of France?", &model)
        .await
        .unwrap();

    assert_eq!(output, EmptyNamedStruct {});
}

/// A named struct
#[derive(Parse, Schema, Clone)]
struct NamedStruct {
    #[parse(rename = "field name")]
    name: String,
    /// The age of the person
    age: u32,
}

#[test]
fn named_struct_schema()  {
    let schema = NamedStruct::schema();
    let json = serde_json::from_str::<serde_json::Value>(&schema.to_string()).unwrap();
    assert_eq!(json, serde_json::json!({
        "title": "NamedStruct",
        "description": "A named struct",
        "properties": {
            "field name": {
                "type": "string"
            },
            "age": {
                "description": "The age of the person",
                "type": "integer"
            }
        },
        "required": [
            "field name",
            "age"
        ]
    }));
}

#[tokio::test]
async fn named_struct() {
    let model = Llama::builder()
        .with_source(LlamaSource::tiny_llama_1_1b_chat())
        .build()
        .await
        .unwrap();

    let task = Task::builder("You generate json")
        .with_constraints(NamedStruct::new_parser())
        .build();

    let output = task
        .run("What is the capital of France?", &model)
        .all_text()
        .await;
    println!("{output}");

    assert!(output.contains("\"field name\":"));
    assert!(output.contains("\"age\":"));
}

#[derive(Parse, Schema, Clone)]
struct WithStruct {
    #[parse(with = StringParser::new(1..=10))]
    name: String,
    #[parse(rename = "field name")]
    age: u32,
}

#[tokio::test]
async fn with_struct() {
    let model = Llama::builder()
        .with_source(LlamaSource::tiny_llama_1_1b_chat())
        .build()
        .await
        .unwrap();

    let task = Task::builder("You generate json")
        .with_constraints(WithStruct::new_parser())
        .build();

    let output = task
        .run("What is the capital of France?", &model)
        .all_text()
        .await;
    println!("{output}");

    assert!(output.contains("\"name\":"));
    assert!(output.contains("\"field name\":"));
}
