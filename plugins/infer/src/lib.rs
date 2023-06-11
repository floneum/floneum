#![allow(unused)]

use core::panic;
use std::vec;

use rust_adapter::*;

export_plugin_world!(Plugin);

pub struct Plugin;

impl Definitions for Plugin {
    fn structure() -> Definition {
        Definition {
            name: "inference".to_string(),
            description: "loads a model and runs it".to_string(),
            inputs: vec![IoDefinition {
                name: "input".to_string(),
                ty: ValueType::Single(PrimitiveValueType::Text),
            }],
            outputs: vec![IoDefinition {
                name: "output".to_string(),
                ty: ValueType::Single(PrimitiveValueType::Text),
            }],
        }
    }

    fn run(input: Vec<Value>) -> Vec<Value> {
        let model = ModelType::Llama(LlamaType::Vicuna);

        let session = ModelInstance::new(model);

        let text_input = match &input[0] {
            Value::Single(PrimitiveValue::Text(text)) => text,
            _ => panic!("expected text input"),
        };

        let mut responce = session.infer(&text_input, Some(100), None);
        responce += "\n";

        print(&responce);

        vec![Value::Single(PrimitiveValue::Text(responce))]
    }
}
