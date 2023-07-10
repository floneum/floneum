use floneum_rust::*;

#[export_plugin(("true", "false"))]
/// Switch between two values based on a condition
fn if_statement(
    value: PrimitiveValue,
    first: bool,
) -> (Option<PrimitiveValue>, Option<PrimitiveValue>) {
    if first {
        (Some(value), None)
    } else {
        (None, Some(value))
    }
}
