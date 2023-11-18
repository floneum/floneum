#![allow(unused)]

use floneum_rust::*;
use nipper::Document;

#[export_plugin]
/// Searches wikipedia, fetches the top article from wikipedia, and returns it as text
fn search_engine(query: String) -> String {
    let url = format!(
        "https://en.wikipedia.org/w/index.php?search={}",
        query.replace(' ', "+")
    );
    let page = Page::new(BrowserMode::Headless, &url);

    let document = Document::from(&page.html());
    let mut results = String::new();
    let mut article_count = 0;

    document.select("a").iter().for_each(|link| {
        if let Some(href) = link.attr("href") {
            if href.starts_with("https://en.wikipedia.org/wiki/") || href.starts_with("/wiki/") {
                if article_count > 5 {
                    return;
                }
                let href = if href.starts_with('/') {
                    format!("https://en.wikipedia.org{}", href)
                } else {
                    href.to_string()
                };
                let request = Page::new(BrowserMode::Headless, &href);
                let html = request.html();

                Document::from(&html)
                    .select("p")
                    .iter()
                    .for_each(|paragragh| {
                        let html = paragragh.text();
                        results += &html;
                        results += "\n";
                    });
                article_count += 1;
            }
        }
    });

    results
}
