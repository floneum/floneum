use floneum_rust::*;

#[export_plugin]
/// Creates a browser tab.
/// 
/// ### Examples
/// vec![
///     Example {
///         inputs: vec![true.into_input_value()],
///         outputs: vec![TabId { id: 0 }.into_return_value()]
///     },
/// ]
fn create_browser(
    /// If the tab should be opened in a headless browser.
    headless: bool,
) -> TabId {
    Tab::new(headless).leak()
}
