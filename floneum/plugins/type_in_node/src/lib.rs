use floneum_rust::*;

#[export_plugin]
/// Types some text in a node
fn type_in_node(
    /// The node to type in
    node: Node,
    /// The text to type
    text: String,
) -> Node {
    node.send_keys(&text);
    node
}
