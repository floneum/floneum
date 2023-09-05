use floneum_rust::*;

#[export_plugin]
/// Multiplies two numbers
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![3.into_input_value(), 2.into_input_value()],
///         outputs: vec![6.into_return_value()]
///     },
/// ]
fn multiply(
    /// the fist number
    first: i64,
    /// the second number
    second: i64,
) -> i64 {
    first * second
}
