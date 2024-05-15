use scraper::{ElementRef, Node};
use kalosm::language::*;
use scraper::Html;
use std::future::Future;
use std::io::Write;
use std::pin::Pin;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let real_visited = Arc::new(AtomicUsize::new(0));
    Page::crawl(
        Url::parse("https://floneum.com").unwrap(),
        BrowserMode::Static,
        move |page: Page| {
            let real_visited = real_visited.clone();
            Box::pin(async move {
                let visited = real_visited.fetch_add(1, Ordering::SeqCst);

                if page.url().domain() != Some("floneum.com") {
                    println!("skipping {:?}", page.url());
                    return CrawlFeedback::follow_none();
                }

                let Ok(document) = page.html().await else {
                    println!("failed to get article {:?}", page.url());
                    return CrawlFeedback::follow_none();
                };

                let simplified = clean_html(document);

                // write the page to disk
                let _ = std::fs::create_dir_all("scraped");
                if let Ok(mut file) = std::fs::File::create(format!("scraped/{visited}.html")){
                    _ = file.write_all(simplified.as_bytes());
                }
                
                CrawlFeedback::follow_all()
            }) as Pin<Box<dyn Future<Output = CrawlFeedback>>>
        },
    )
    .await
    .unwrap();
}

const IMPORTANT_ATTRIBUTES: &[&str] = &["id", "href", "alt", "title", "role", "type", "src"];
const IGNORE_ELEMENTS: &[&str] = &["script", "style"];
const IMPORTANT_ELEMENTS: &[&str] = &[
    "a", "img", "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li",
    "dl", "dt", "dd",
    "table", "tr", "td", "th",
    "select", "option", "form", "label",
];

fn clean_html(html: Html) -> String {
    let mut result = String::new();

    visit_element(html.root_element(), &mut result, &mut false);

    result
}

fn visit_element(element: ElementRef, result: &mut String, last_node_text: &mut bool) {
    let value = element.value();
    let lowercase_name = value.name().to_lowercase();

    if IGNORE_ELEMENTS.contains(&lowercase_name.as_str()) {
        return;
    }

    let is_important = IMPORTANT_ELEMENTS.contains(&lowercase_name.as_str());
    if is_important {
        result.push('<');
        result.push_str(&lowercase_name);

        for (attribute, value) in value.attrs() {
            if IMPORTANT_ATTRIBUTES.contains(&attribute) {
                result.push(' ');
                result.push_str(attribute);
                result.push('=');
                result.push('"');
                result.push_str(value);
                result.push('"');
            }
        }

        result.push('>');
        *last_node_text = false;
    }

    for child in element.children() {
        match child.value() {
            Node::Element(_) => {
                visit_element(ElementRef::wrap(child).unwrap(), result, last_node_text);
            },
            Node::Text(t) => {
                let trimmed = t.trim();
                if !trimmed.is_empty() {
                    if *last_node_text {
                        result.push(' ');
                    }
                    *last_node_text = true;
                    result.push_str(trimmed);
                }
            },
            _ => {}
        }
    }

    if is_important {
        result.push('<');
        result.push('/');
        result.push_str(&lowercase_name);
        result.push('>');
        *last_node_text = false;
    }
}
