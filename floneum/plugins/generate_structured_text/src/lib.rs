#![allow(unused)]

use core::panic;
use std::vec;

use floneum_rust::*;

#[export_plugin]
/// Calls a large language model to generate structured text. You can create a template for the language model to fill in. The model will fill in any segments that contain {**type**} where **type** is "", bool, or #
///
/// It is important to keep in mind that the language model is just generating text. Because the model is merely continuing the text you give it, the formatting of that text can be important.
///
/// It is commonly helpful to provide a few examples to the model before your new data so that the model can pick up on the pattern of the text
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![ModelType::LlamaSevenChat.into_input_value(), String::from("10x10=").into_input_value(), String::from(r"\d{3}").into_input_value()],
///         outputs: vec![String::from("10").into_return_value()],
///     },
/// ]
fn generate_structured_text(
    /// the model to use
    model: ModelType,
    /// the prompt to use when running the model
    prompt: String,
    /// the structure the model output will follow
    regex: String,
) -> String {
    if !TextGenerationModel::model_downloaded(model) {
        log_to_user("downloading model... This could take several minutes");
    }

    let session = TextGenerationModel::new(model);

    let mut responce = session.infer_structured(&prompt, &regex);
    responce += "\n";

    responce
}
