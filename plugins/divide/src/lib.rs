use floneum_rust::*;

#[export_plugin]
/// Divide one number by another
/// 
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![2.into_input_value(), 2.into_input_value()],
///         outputs: vec![1.into_return_value()]
///     },
/// ]
fn add(
    /// the fist number
    first: i64,
    /// the second number
    second: i64,
) -> i64 {
    first / second
}
