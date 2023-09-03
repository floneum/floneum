use floneum_rust::*;

#[export_plugin]
/// Types some text in a node
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![NodeId { id: 0, tab: TabId { id: 0 } }.into_input_value(), String::from("Hello World!").into_input_value()],
///         outputs: vec![NodeId { id: 0, tab: TabId { id: 0 } }.into_return_value()],
///     },
/// ]
fn type_in_node(
    /// The node to type in
    node: Node,
    /// The text to type
    text: String,
) -> Node {
    node.type_text(&text);
    node
}
