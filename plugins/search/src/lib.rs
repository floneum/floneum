use floneum_rust::*;

#[export_plugin]
/// searches an embedding database for the closest embedding
pub fn search(
    /// the embedding to search for
    key: Embedding,
    /// the embedding database to search
    database: EmbeddingDbId,
) -> String {
    let database = VectorDatabase::from_id(database);
    let nearest = database.find_closest_documents(&key, 5);
    println!("nearest: {:?}\n", nearest);

    let mut message = String::new();
    for (i, embedding) in nearest.iter().enumerate() {
        message.push_str(&format!("{}: {}\n", i, embedding));
    }
    message
}
