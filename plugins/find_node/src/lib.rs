use floneum_rust::*;

#[export_plugin]
/// Finds a node in a tab
fn find_node(
    /// The tab to find the element in
    tab: Tab,
    /// The selector to find the element with
    selector: String,
) -> Tab {
    tab.wait_for_element(&selector);
    tab
}
