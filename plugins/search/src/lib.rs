use floneum_rust::*;

#[export_plugin]
/// Searches an embedding database for the closest embedding
/// 
/// This node requires that the Embedding Database and the Embedding use the same model.
/// 
/// Returns text with documents separated with newlines.
pub fn search(
    /// the embedding to search for
    key: Embedding,
    /// the embedding database to search
    database: EmbeddingDbId,
    /// the number of documents to return
    top_n: i64,
) -> String {
    let database = VectorDatabase::from_id(database);
    let nearest = database.find_closest_documents(&key, top_n.abs().try_into().unwrap());
    println!("nearest: {:?}\n", nearest);

    let mut message = String::new();
    for embedding in &nearest {
        message.push_str(&format!("{}\n", embedding));
    }
    message
}
