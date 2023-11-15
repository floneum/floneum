use floneum_rust::*;

#[export_plugin]
/// Check if either input is true
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![true.into_input_value(), false.into_input_value()],
///         outputs: vec![true.into_return_value()]
///     },
/// ]
fn or(
    /// the first boolean
    first: bool,
    /// the second boolean
    second: bool,
) -> bool {
    first || second
}
