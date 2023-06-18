use rust_adapter::*;

#[export_plugin]
/// searches an embedding database for the closest embedding
pub fn search(
    /// the embedding to search for
    key: Embedding,
    /// the embedding database to search
    database: EmbeddingDbId,
) -> String {
    let database = VectorDatabase::from_id(database);
    let nearest = database.find_closest_documents(&key, 1);

    nearest.first().cloned().unwrap_or_default()
}
