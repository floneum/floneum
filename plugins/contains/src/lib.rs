use floneum_rust::*;

#[export_plugin]
/// Checks if some text contains some other text. Returns true if the first text contains the second text.
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![String::from("Hello World").into_input_value(), String::from("World").into_input_value()],
///         outputs: vec![true.into_return_value()],
///     },
/// ]
fn contains(value: String, contains: String) -> bool {
    value.contains(&contains)
}
