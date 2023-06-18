use rust_adapter::*;

#[export_plugin]
/// creates embeddings for text
fn embedding(input: String) -> Embedding {
    let model = ModelInstance::new(ModelType::Llama(LlamaType::Vicuna));

    let embedding = model.get_embedding(&input);

    embedding
}
