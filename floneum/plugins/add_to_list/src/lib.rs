use floneum_rust::*;

#[export_plugin]
/// Add an element to a list
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![vec![String::from("Text")].into_input_value(), String::from("New text").into_input_value()],
///         outputs: vec![vec![String::from("Text"), String::from("New text")].into_return_value()]
///     },
/// ]
fn add_to_list(
    /// the list to add to
    list: Vec<String>,
    /// the element to add
    element: String,
) -> Vec<String> {
    let mut list = list;
    list.push(element);
    list
}
