use floneum_rust::*;

#[export_plugin]
/// Clicks a node in a tab
/// 
/// ### Examples
/// vec![
///     Example {
///         inputs: vec![NodeId { id: 0, tab: TabId { id: 0 } }.into_input_value()],
///         outputs: vec![NodeId { id: 0, tab: TabId { id: 0 } }.into_return_value()],
///     },
/// ]
fn click_node(
    /// The node to click
    node: Node,
) -> Node {
    node.click();
    node
}
