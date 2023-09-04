use floneum_rust::*;

#[export_plugin]
/// Takes a string and returns the string
/// 
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![String::from("hello").into_input_value()],
///         outputs: vec![String::from("hello").into_return_value()]
///     },
/// ]
fn string(
    /// the string
    string: String,
) -> String {
    string
}
