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
                    ty: ValueType::Text,
                },
                IoDefinition {
                    name: "input".to_string(),
                    ty: ValueType::Text,
                },
            ],
            outputs: vec![IoDefinition {
                name: "output".to_string(),
                ty: ValueType::Text,
            }],
        }
    }

    fn run(input: Vec<Value>) -> Vec<Value> {
        let template = match &input[0] {
            Value::Text(text) => text,
            _ => panic!("expected text input"),
        };

        let input = match &input[1] {
            Value::Text(text) => text,
            _ => panic!("expected text input"),
        };

        let text = template.replacen("{}", input, 1);

        vec![Value::Text(text)]
    }
}
