use floneum_rust::*;

#[export_plugin]
/// Create an empty list
/// 
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![],
///         outputs: vec![Vec::<String>::new().into_return_value()]
///     },
/// ]
fn new_list() -> Vec<String> {
    Vec::<String>::new()
}
