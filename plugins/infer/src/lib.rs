#![allow(unused)]

use floneum_rust::*;

#[export_plugin]
/// loads a model and runs it
fn inference(input: String) -> String {
    let model = ModelType::Llama(LlamaType::Orca);

    let session = ModelInstance::new(model);

    let mut responce = session.infer(&input, Some(100), None);
    responce += "\n";

    responce
}
