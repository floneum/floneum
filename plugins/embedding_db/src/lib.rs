use floneum_rust::*;

#[export_plugin]
/// Creates a database of embeddings. (A database is just a different way to store information, in this cases this stores documents in a way that makes it easy to find other documents with similar meanings)
///
/// When using this embedding database, you must use the same model to generate the embeddings you insert into this database.
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
