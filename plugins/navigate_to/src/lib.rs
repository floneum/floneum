use floneum_rust::*;

#[export_plugin]
/// Navigate a tab to a URL
/// 
/// ### Examples
/// vec![
///     Example {
///         inputs: vec![TabId { id: 0 }.into_input_value(), String::from("https://floneum.com").into_input_value()],
///         outputs: vec![TabId { id: 0 }.into_return_value()],
///     },
/// ]
fn navigate_to(
    /// The tab to navigate
    tab: Tab,
    /// The URL to navigate to
    url: String,
) -> Tab {
    if url.starts_with("http://") || url.starts_with("https://") {
        tab.goto(&url);
    } else {
        tab.goto(&format!("http://{}", url));
    }
    tab
}
