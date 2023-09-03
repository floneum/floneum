use floneum_rust::*;

#[export_plugin]
/// Join multiple strings into one string separated by a separator
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![vec![String::from("Text to join"), String::from("Another text to join")].into_input_value(), String::from(",").into_input_value()],
///         outputs: vec![String::from("Text to join,Another text to join").into_return_value()]
///     },
/// ]
fn join(
    /// the list of strings to join
    text: Vec<String>,
    /// the separator to join by
    separator: String,
) -> String {
    text.join(&separator)
}
