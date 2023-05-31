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
        let models = [
            ModelType::Llama(LlamaType::Vicuna),
            ModelType::Llama(LlamaType::Guanaco),
            ModelType::GptNeoX(GptNeoXType::DollySevenB),
            ModelType::GptNeoX(GptNeoXType::TinyPythia),
            ModelType::GptNeoX(GptNeoXType::LargePythia),
            ModelType::Mpt(MptType::Chat),
            ModelType::Mpt(MptType::Story),
            ModelType::Mpt(MptType::Instruct),
            ModelType::Mpt(MptType::Base),
        ];

        let mut outputs = String::new();

        for model in models {
            let session = load_model(model);

            let text_input = match &input[0] {
                Value::Text(text) => text,
                _ => panic!("expected text input"),
            };

            let text_input = format!("This is a chat between an AI chatbot and a human. The chatbot is programmed to be extremly helpful and always attempt to answer correctly. The human will start questions with ### Human; the AI with start answers with ### Assistant\n### Human{text_input}\n ### Assistant");

            let responce = infer(session, &text_input, Some(50), Some("### Human"));

            let text = format!("{model:?}:\n{}\n\n", responce);
            print(&text);
            outputs += &text;

            unload_model(session);
        }

        vec![Value::Text(outputs)]
    }
}
