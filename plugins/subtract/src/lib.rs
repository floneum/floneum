use floneum_rust::*;

#[export_plugin]
/// Subtract the second number from the first number
/// 
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![1.into_input_value(), 2.into_input_value()],
///         outputs: vec![(-1i64).into_return_value()]
///     },
/// ]
fn add(
    /// the fist number
    first: i64,
    /// the second number
    second: i64,
) -> i64 {
    first - second
}
