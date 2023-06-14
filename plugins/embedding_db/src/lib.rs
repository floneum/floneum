use rust_adapter::*;

export_plugin_world!(Plugin);

pub struct Plugin;

impl Definitions for Plugin {
    fn structure() -> Definition {
        Definition {
            name: "embedding data base".to_string(),
            description: "a database for embeddings".to_string(),
            inputs: vec![
                IoDefinition {
                    name: "text".to_string(),
                    ty: ValueType::Single(PrimitiveValueType::Text),
                },
                IoDefinition {
                    name: "seperator".to_string(),
                    ty: ValueType::Single(PrimitiveValueType::Text),
                },
            ],
            outputs: vec![IoDefinition {
                name: "database".to_string(),
                ty: ValueType::Single(PrimitiveValueType::Database),
            }],
        }
    }

    fn run(input: Vec<Value>) -> Vec<Value> {
        let text = match &input[0] {
            Value::Single(PrimitiveValue::Text(text)) => text,
            _ => panic!("expected text input"),
        };
        let seperator = match &input[1] {
            Value::Single(PrimitiveValue::Text(text)) => text,
            _ => panic!("expected text input"),
        };

        let model = ModelType::Llama(LlamaType::Vicuna);
        let instance = ModelInstance::new(model);

        let borrowed_documents = text.split(seperator).collect::<Vec<_>>();
        let embeddings = borrowed_documents
            .iter()
            .map(|s| instance.get_embedding(s))
            .collect::<Vec<_>>();
        let borrowed_embeddings = embeddings.iter().collect::<Vec<_>>();

        let database = VectorDatabase::new(&borrowed_embeddings, &borrowed_documents);

        vec![Value::Single(PrimitiveValue::Database(database.leak()))]
    }
}
