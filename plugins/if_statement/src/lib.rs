use floneum_rust::*;

#[export_plugin(("true", "false"))]
/// switch between two values based on a condition
fn if_statement(value: String, first: bool) -> (Option<String>, Option<String>) {
    if first {
        (Some(value), None)
    } else {
        (None, Some(value))
    }
}
