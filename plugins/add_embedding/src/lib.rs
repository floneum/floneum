use rust_adapter::*;

#[export_plugin]
/// creates a database of embeddings
fn add_embedding(
    /// the database to add the embedding to
    database: EmbeddingDbId,
    /// the embedding to add
    embedding: Embedding,
    /// the value to associate with the embedding
    value: String,
) {
    let database = VectorDatabase::from_id(database);
    database.add_embedding(&embedding, &value);
}
