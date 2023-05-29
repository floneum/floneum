wit_bindgen::generate!(in "../wit");
use crate::plugins::main::imports::*;

struct Plugin;

export_plugin_world!(Plugin);

impl PluginWorld for Plugin {
    fn start() {
        let session = load_model(ModelType::Mpt(MptType::Instruct));

        let responce = infer(
            session,
            "### Assistant: Hello - How may I help you today?",
            " ",
        );

        print(&responce);
    }
}
