use floneum_rust::*;

#[export_plugin]
/// Gets the text of a node
fn click_node(
    /// The node to extract the text from
    node: Node,
) -> String {
    node.text()
}
