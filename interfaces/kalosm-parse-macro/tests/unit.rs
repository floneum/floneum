#![allow(unused)]

use kalosm::language::{kalosm_sample, Parse, Schema};

#[derive(Parse, Schema, Clone)]
struct UnitStruct;

#[cfg(any(feature = "metal", feature = "cuda"))]
#[tokio::test]
async fn unit_struct() {
    use kalosm::language::*;

    let model = Llama::builder()
        .with_source(LlamaSource::tiny_llama_1_1b_chat())
        .build()
        .await
        .unwrap();

    let task = model
        .task("You generate json")
        .with_constraints(std::sync::Arc::new(UnitStruct::new_parser()));

    let output = task.run("What is the capital of France?", &model).await;
    println!("{output}");

    assert_eq!(output, "\"UnitStruct\"");
}

#[derive(Parse, Schema, Clone)]
#[parse(rename = "unit struct")]
struct RenamedUnit;

#[cfg(any(feature = "metal", feature = "cuda"))]
#[tokio::test]
async fn renamed_unit_struct() {
    use kalosm::language::*;

    let model = Llama::builder()
        .with_source(LlamaSource::tiny_llama_1_1b_chat())
        .build()
        .await
        .unwrap();

    let task = model
        .task("You generate json")
        .with_constraints(std::sync::Arc::new(RenamedUnit::new_parser()));

    let output = task.run("What is the capital of France?", &model).await;
    println!("{output}");

    assert_eq!(output, "\"unit struct\"");
}
