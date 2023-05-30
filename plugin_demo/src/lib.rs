wit_bindgen::generate!(in "../wit");
use crate::plugins::main::imports::*;

struct Plugin;

export_plugin_world!(Plugin);

impl PluginWorld for Plugin {
    fn start() {
        for model in [
            ModelType::GptNeoX(GptNeoXType::TinyPythia),
            ModelType::GptNeoX(GptNeoXType::LargePythia),
            ModelType::Llama(LlamaType::Vicuna),
            ModelType::GptNeoX(GptNeoXType::DollySevenB),
            ModelType::Mpt(MptType::Instruct),
            ModelType::Mpt(MptType::Chat),
            ModelType::Mpt(MptType::Base),
            ModelType::Mpt(MptType::Story),
        ] {
            let session = load_model(model);

            let responce = infer(
                session,
                r####"A chat between a human and an assistant. The assistant is trained to be helpful and respond with a responce starting with "### Assistant".
### Human
What are the most important countries in Europe geopolitically?
### Assistant
"####,
                Some(500),
                Some("### Human"),
            );

            print(&(responce + "\n\n\n"));

            unload_model(session);
        }
    }
}
