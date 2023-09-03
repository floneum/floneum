use floneum_rust::*;

#[export_plugin]
/// Return the opposite of the input
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![true.into_input_value()],
///         outputs: vec![false.into_return_value()]
///     },
/// ]
fn not(
    /// the boolean value
    value: bool,
) -> bool {
    !value
}
