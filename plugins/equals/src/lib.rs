use floneum_rust::*;

#[export_plugin]
/// Check if a number is equal to another number
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![1.into_input_value(), 2.into_input_value()],
///         outputs: vec![false.into_return_value()]
///     },
/// ]
fn equals(
    /// the first number
    first: i64,
    /// the second number
    second: i64,
) -> bool {
    first == second
}
