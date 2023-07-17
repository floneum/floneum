use floneum_rust::*;

#[export_plugin]
/// Creates a browser tab.
fn create_browser(
    /// If the tab should be opened in a headless browser.
    headless: bool,
) -> Tab {
    Tab::new(headless)
}
