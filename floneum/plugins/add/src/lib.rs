use floneum_rust::*;

#[export_plugin]
/// Adds two numbers
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![1.into_input_value(), 2.into_input_value()],
///         outputs: vec![3.into_return_value()]
///     },
/// ]
fn add(
    /// the fist number
    first: i64,
    /// the second number
    second: i64,
) -> i64 {
    first + second
}
