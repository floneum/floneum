use floneum_rust::*;

#[export_plugin]
/// Searches an embedding database for the closest embedding
///
/// This node requires that the Embedding Database and the Embedding use the same model.
///
/// Returns text with documents separated with newlines.
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![Embedding { vector: vec![0.0, 0.0, 0.0] }.into_input_value(), EmbeddingDbId { id: 0 }.into_input_value(), 10.into_input_value()],
///         outputs: vec![String::from("Document 1\nDocument 2\nDocument 3\nDocument 4\nDocument 5\nDocument 6\nDocument 7\nDocument 8\nDocument 9\nDocument 10\n").into_return_value()],
///     },
/// ]
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

    let mut message = String::new();
    for embedding in &nearest {
        message.push_str(&format!("{}\n", embedding));
    }
    message
}
