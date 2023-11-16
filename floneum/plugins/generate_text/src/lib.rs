#![allow(unused)]

use floneum_rust::{plugins::main::imports::log_to_user, *};

#[export_plugin]
/// Calls a large language model to generate text.
///
/// It is important to keep in mind that the language model is just generating text. Because the model is merely continuing the text you give it, the formatting of that text can be important.
///
/// It is commonly helpful to provide a few examples to the model before your new data so that the model can pick up on the pattern of the text
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![ModelType::Llama(LlamaType::LlamaSevenChat).into_input_value(), String::from("The following is a chat between a user and an assistant. The assistant helpfully and succinctly answers questions posed by the user.\n### USER\nWhat is 3 + 6?\n### ASSISTANT\n9\n### USER\nWhat is 1 + 1?\n### ASSISTANT\n2\n### USER\nWhat is 2 + 8?\n### ASSISTANT\n{#}").into_input_value()],
///         outputs: vec![String::from("ten").into_return_value()],
///     },
/// ]
fn generate_text(model: ModelType, text: String, max_size: i64) -> String {
    if !ModelType::model_downloaded(model) {
        log_to_user("downloading model... This could take several minutes");
    }

    let session = Model::new(model);

    let mut responce = session.infer(&text, (max_size != 0).then(|| max_size as u32), None);
    responce += "\n";

    responce
}
