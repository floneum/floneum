use rust_adapter::*;

export_plugin_world!(Plugin);

pub struct Plugin;

impl Definitions for Plugin {
    fn structure() -> Definition {
        Definition {
            name: "embedding".to_string(),
            description: "creates embeddings for text".to_string(),
            inputs: vec![IoDefinition {
                name: "input".to_string(),
                ty: ValueType::Single(PrimitiveValueType::Text),
            }],
            outputs: vec![IoDefinition {
                name: "embedding".to_string(),
                ty: ValueType::Single(PrimitiveValueType::Embedding),
            }],
        }
    }

    fn run(input: Vec<Value>) -> Vec<Value> {
        let input = match &input[0] {
            Value::Single(PrimitiveValue::Text(text)) => text,
            _ => panic!("expected text input"),
        };

        let model = ModelInstance::new(ModelType::Llama(LlamaType::Vicuna));

        let embedding = model.get_embedding(input);

        vec![Value::Single(PrimitiveValue::Embedding(embedding))]
    }
}
