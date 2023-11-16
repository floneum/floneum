use floneum_rust::*;

#[export_plugin]
/// Gets the text of a node
fn node_text(
    /// The node to extract the text from
    node: Node,
) -> (String, Node) {
    (node.get_element_text(), node)
}
