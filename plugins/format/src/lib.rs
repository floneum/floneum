use rust_adapter::*;

export_plugin_world!(Plugin);

pub struct Plugin;

impl Definitions for Plugin {
    fn structure() -> Definition {
        Definition {
            name: "format".to_string(),
            description: "formats text with a template".to_string(),
            inputs: vec![
                IoDefinition {
                    name: "template".to_string(),
                    ty: ValueType::Single(PrimitiveValueType::Text),
                },
                IoDefinition {
                    name: "input".to_string(),
                    ty: ValueType::Single(PrimitiveValueType::Text),
                },
            ],
            outputs: vec![IoDefinition {
                name: "output".to_string(),
                ty: ValueType::Single(PrimitiveValueType::Text),
            }],
        }
    }

    fn run(input: Vec<Value>) -> Vec<Value> {
        let template = match &input[0] {
            Value::Single(PrimitiveValue::Text(text)) => text,
            _ => panic!("expected text input"),
        };

        let input = match &input[1] {
            Value::Single(PrimitiveValue::Text(text)) => text,
            _ => panic!("expected text input"),
        };

        let text = template.replacen("{}", input, 1);

        vec![Value::Single(PrimitiveValue::Text(text))]
    }
}
