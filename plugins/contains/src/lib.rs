use floneum_rust::*;

#[export_plugin]
/// check if a string contains another string
fn contains(value: String, contains: String) -> bool {
    value.contains(&contains)
}
