use floneum_rust::*;

#[export_plugin]
/// Navigate a tab to a URL
fn navigate_to(
    /// The tab to navigate
    tab: Tab,
    /// The URL to navigate to
    url: String,
) {
    tab.goto(&url)
}
