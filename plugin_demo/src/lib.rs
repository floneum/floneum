wit_bindgen::generate!(in "../wit");
use crate::exports::plugins::main::definitions::*;
use crate::plugins::main::imports::*;

struct Plugin;

export_plugin_world!(Plugin);

impl Definitions for Plugin {
    fn structure() -> Definition {
        Definition {
            name: "test plugin".to_string(),
            description: "this is a test plugin".to_string(),
            inputs: vec![],
        }
    }
}

impl PluginWorld for Plugin {
    fn start() {
        let model = ModelType::GptNeoX(GptNeoXType::TinyPythia);
        let session = load_model(model);

        let responce = infer(
            session,
            r####"A chat between a human and an assistant. The assistant is trained to be helpful and respond with a responce starting with "### Assistant".
### Human
What is the capital of France?
### Assistant
"####,
            Some(50),
            Some("### Human"),
        );

        print(&(responce + "\n\n\n"));
    }
}
