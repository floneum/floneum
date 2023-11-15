use floneum_rust::*;

#[export_plugin]
/// Split a string into multiple strings separated by a separator
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![String::from("Text to split,Another text to split").into_input_value(), String::from(",").into_input_value()],
///         outputs: vec![vec![String::from("Text to split"), String::from("Another text to split")].into_return_value()]
///     },
/// ]
fn split(
    /// the text to split
    text: String,
    /// the separator to split by
    separator: String,
) -> Vec<String> {
    text.split(&separator)
        .filter(|text| !text.is_empty())
        .map(|text| text.to_string())
        .collect::<Vec<_>>()
}
