use floneum_rust::*;

#[export_plugin]
/// Check if both inputs are true
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![true.into_input_value(), true.into_input_value()],
///         outputs: vec![true.into_return_value()]
///     },
/// ]
fn and(
    /// the first boolean
    first: bool,
    /// the second boolean
    second: bool,
) -> bool {
    first && second
}
