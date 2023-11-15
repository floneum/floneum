use floneum_rust::*;

#[export_plugin]
/// Finds a node in a tab
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![TabId { id: 0 }.into_input_value(), String::from("body").into_input_value()],
///         outputs: vec![NodeId { id: 0, tab: TabId { id: 0 } }.into_return_value()],
///     },
/// ]
fn find_child_node(
    /// The tab to find the element in
    node: Node,
    /// The selector to find the element with
    selector: String,
) -> Node {
    node.wait_for_element(&selector)
}
