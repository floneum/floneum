use floneum_rust::*;

#[export_plugin]
/// creates a database of embeddings
fn embedding_db(
    /// the model to use
    model: ModelType,
    /// the separator between documents
    separator: String,
    /// the documents to index
    text: String,
) -> EmbeddingDbId {
    let instance = ModelInstance::new(model);

    let borrowed_documents = text
        .split(&separator)
        .filter(|text| !text.is_empty())
        .map(|text| text.to_string())
        .collect::<Vec<_>>();
    let embeddings = borrowed_documents
        .iter()
        .map(|s| instance.get_embedding(s))
        .collect::<Vec<_>>();

    let database = VectorDatabase::new(&embeddings, &borrowed_documents);

    database.leak()
}
