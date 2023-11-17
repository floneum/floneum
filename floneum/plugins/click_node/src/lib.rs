use floneum_rust::*;

#[export_plugin]
/// Clicks a node in a tab
fn click_node(
    /// The node to click
    node: Node,
) -> Node {
    node.click_element();
    node
}
