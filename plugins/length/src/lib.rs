use floneum_rust::*;

#[export_plugin]
/// Get the number of elements in a list
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![vec![String::from("Text"), String::from("Another text")].into_input_value()],
///         outputs: vec![2.into_return_value()]
///     },
/// ]
fn length(
    /// the list to get the length of
    list: Vec<String>,
) -> i64 {
    list.len() as i64
}
