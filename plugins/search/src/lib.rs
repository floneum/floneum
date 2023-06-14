use rust_adapter::*;

export_plugin_world!(Plugin);

pub struct Plugin;

impl Definitions for Plugin {
    fn structure() -> Definition {
        Definition {
            name: "search".to_string(),
            description: "searches an embedding database for the closest embedding".to_string(),
            inputs: vec![
                IoDefinition {
                    name: "input".to_string(),
                    ty: ValueType::Single(PrimitiveValueType::Embedding),
                },
                IoDefinition {
                    name: "embedding database".to_string(),
                    ty: ValueType::Single(PrimitiveValueType::Database),
                },
            ],
            outputs: vec![IoDefinition {
                name: "nearest".to_string(),
                ty: ValueType::Single(PrimitiveValueType::Text),
            }],
        }
    }

    fn run(input: Vec<Value>) -> Vec<Value> {
        let search = match &input[0] {
            Value::Single(PrimitiveValue::Embedding(embedding)) => embedding,
            _ => panic!("expected embedding input"),
        };
        let database_id = match &input[1] {
            Value::Single(PrimitiveValue::Database(database)) => database,
            _ => panic!("expected database input"),
        };
        let database = VectorDatabase::from_id(*database_id);

        let nearest = database.find_closest_documents(&search, 1);

        vec![Value::Single(PrimitiveValue::Text(
            nearest.first().cloned().unwrap_or_default(),
        ))]
    }
}
