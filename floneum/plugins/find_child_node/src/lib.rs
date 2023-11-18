use floneum_rust::*;

#[export_plugin]
/// Finds a node in a tab
fn find_child_node(
    /// The tab to find the element in
    node: Node,
    /// The selector to find the element with
    selector: String,
) -> Node {
    node.find_child_of_element(&selector)
}
