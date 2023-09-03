use floneum_rust::*;

#[export_plugin]
/// Gets the text of a node
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![NodeId { id: 0, tab: TabId { id: 0 } }.into_input_value()],
///         outputs: vec![String::from("Node Text Content").into_return_value(), NodeId { id: 0, tab: TabId { id: 0 } }.into_return_value()],
///     },
/// ]
fn node_text(
    /// The node to extract the text from
    node: Node,
) -> (String, Node) {
    (node.text(), node)
}
