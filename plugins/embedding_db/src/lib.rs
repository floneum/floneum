use floneum_rust::*;

#[export_plugin]
/// creates a database of embeddings
fn embedding_db(
    /// the seperator between documents
    seperator: String,
    /// the documents to index
    text: String,
) -> EmbeddingDbId {
    let model = ModelType::Llama(LlamaType::Vicuna);
    let instance = ModelInstance::new(model);

    let borrowed_documents = text
        .split(&seperator)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>();
    let embeddings = borrowed_documents
        .iter()
        .map(|s| instance.get_embedding(s))
        .collect::<Vec<_>>();
    let borrowed_embeddings = embeddings.iter().collect::<Vec<_>>();

    let database = VectorDatabase::new(&borrowed_embeddings, &borrowed_documents);

    database.leak()
}
