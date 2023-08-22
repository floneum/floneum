use floneum_rust::*;
use url::Url;

#[export_plugin]
/// Read an article from a URL
fn get_article(
    /// The article URL
    url: String,
) -> String {
    let base_url = Url::parse(&url).unwrap();
    let html = get_request(&url, &[]);
    let cleaned = readability::extractor::extract(&mut html.as_bytes(), &base_url).unwrap();
    cleaned.text
}
