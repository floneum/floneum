use floneum_rust::*;

#[export_plugin]
/// Checks if some text contains some other text. Returns true if the first text contains the second text.
fn contains(value: String, contains: String) -> bool {
    value.contains(&contains)
}
