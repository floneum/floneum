use floneum_rust::*;

#[export_plugin]
/// Navigate a tab to a URL
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![String::from("https://floneum.com").into_input_value()],
///         outputs: vec![Page::new(BrowserMode::Headless, "https://floneum.com").into_return_value()],
///     },
/// ]
fn navigate_to(
    /// The URL to navigate to
    url: String,
) -> Page {
    if url.starts_with("http://") || url.starts_with("https://") {
        Page::new(BrowserMode::Headless, &url)
    } else {
        Page::new(BrowserMode::Headless, &format!("http://{}", url))
    }
}
