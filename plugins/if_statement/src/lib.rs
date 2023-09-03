use floneum_rust::*;

#[export_plugin(("true", "false"))]
/// Switch between two values based on a condition
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![String::from("Some Text").into_input_value(), true.into_input_value()],
///         outputs: vec![String::from("Some Text").into_return_value()],
///     },
/// ]
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
