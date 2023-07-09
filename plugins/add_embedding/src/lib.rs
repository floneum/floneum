use floneum_rust::*;

#[export_plugin]
/// Adds a embedding to a database. The model used to generate the embedding and the model type used to create the database must be the same.
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
