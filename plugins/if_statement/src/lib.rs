use floneum_rust::*;

#[export_plugin]
/// creates embeddings for text
fn embedding(value: String, first: bool) -> (Option<String>, Option<String>) {
    if first {
        (Some(value), None)
    }
    else {
        (None, Some(value))
    }
}
