use rust_adapter::*;

export_plugin_world!(Plugin);

pub struct Plugin;

impl Definitions for Plugin {
    fn structure() -> Definition {
        Definition {
            name: "embedding data base".to_string(),
            description: "a database for embeddings".to_string(),
            inputs: vec![],
            outputs: vec![IoDefinition {
                name: "database".to_string(),
                ty: ValueType::Database,
            }],
        }
    }

    fn run(input: Vec<Value>) -> Vec<Value> {
        let input = match &input[0] {
            Value::Text(text) => text,
            _ => panic!("expected text input"),
        };

        let model =  ModelInstance::new(ModelType::Llama(LlamaType::Vicuna));

        let embedding = model.get_embedding(input);

        vec![Value::Embedding(embedding)]
    }
}
