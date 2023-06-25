use floneum_rust::*;

#[export_plugin]
/// creates embeddings for text
fn embedding(model_type: ModelType, input: String) -> Embedding {
    let model = ModelInstance::new(model_type);

    model.get_embedding(&input)
}
