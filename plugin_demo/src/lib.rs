use std::vec;

use crate::exports::plugins::main::definitions::*;
use crate::plugins::main::imports::*;

wit_bindgen::generate!(in "../wit");

struct Plugin;

export_plugin_world!(Plugin);

impl Definitions for Plugin {
    fn structure() -> Definition {
        Definition {
            name: "test plugin".to_string(),
            description: "this is a test plugin".to_string(),
            inputs: vec![IoDefinition {
                name: "input".to_string(),
                ty: ValueType::Text,
            }],
            outputs: vec![IoDefinition {
                name: "output".to_string(),
                ty: ValueType::Text,
            }],
        }
    }

    fn run(input: Vec<Value>) -> Vec<Value> {
        let model = ModelType::GptNeoX(GptNeoXType::TinyPythia);
        let session = load_model(model);

        let text_input = match &input[0] {
            Value::Text(text) => text,
            _ => panic!("expected text input"),
        };

        let responce = infer(session, text_input, Some(50), Some("### Human"));

        print(&(responce.clone() + "\n\n\n"));

        vec![Value::Text(responce)]
    }
}
