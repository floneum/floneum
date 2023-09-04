use floneum_rust::*;

#[export_plugin]
/// Takes a number and returns the number
/// 
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![1.into_input_value()],
///         outputs: vec![1.into_return_value()]
///     },
/// ]
fn number(
    /// the number
    number: i64,
) -> i64 {
    number
}
