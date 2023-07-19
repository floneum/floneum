#![allow(unused)]

use floneum_rust::{plugins::main::imports::log_to_user, *};

#[export_plugin]
/// Calls a large language model to generate text.
///
/// It is important to keep in mind that the language model is just generating text. Because the model is merely continuing the text you give it, the formatting of that text can be important.
///
/// It is commonly helpful to provide a few examples to the model before your new data so that the model can pick up on the pattern of the text
///
/// Example:
/// The following is a chat between a user and an assistant. The assistant helpfully and succinctly answers questions posed by the user.
/// ### USER
/// Where is Paris.
/// ### ASSISTANT
/// Paris is in France. France is in Europe.
/// ### USER
/// What is 1 + 1?
/// ### ASSISTANT
/// 2
/// ### USER
/// **your real question**
/// ### ASSISTANT
///
fn generate_text(model: ModelType, input: String) -> String {
    if !model_downloaded(model) {
        log_to_user("downloading model... This could take several minutes");
    }

    let session = ModelInstance::new(model);

    let mut responce = session.infer(&input, Some(100), None);
    responce += "\n";

    responce
}
