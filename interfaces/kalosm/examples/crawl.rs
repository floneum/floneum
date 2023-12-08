use kalosm_language::*;
use scraper::ElementRef;
use scraper::Html;
use std::future::Future;
use std::io::Write;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let count = Arc::new(AtomicUsize::new(0));
    let real_visited = Arc::new(AtomicUsize::new(0));
    Page::crawl(
        Url::parse("https://bestoftailwind.com/").unwrap(),
        BrowserMode::Static,
        move |page: Page| {
            let count = count.clone();
            let real_visited = real_visited.clone();
            Box::pin(async move {
                real_visited.fetch_add(1, Ordering::SeqCst);
                let current_count = count.load(Ordering::SeqCst);
                if current_count > 1000 {
                    return CrawlFeedback::Stop;
                }

                let Ok(html) = page.html().await else {
                    return CrawlFeedback::follow_none();
                };

                let heuristic = tailwind_heuristic_html(&html);
                println!("Tailwind heuristic: {:?}", heuristic);
                if heuristic.tailwind() {
                    // Write the html to a file
                    let path = PathBuf::from(format!("html/{}.html", count.load(Ordering::SeqCst)));
                    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                    let mut file = std::fs::File::create(path).unwrap();
                    let mut formatted = String::new();
                    write_element(*html.root_element(), &mut formatted);
                    let _ = file.write_all(formatted.as_bytes());
                    count.fetch_add(1, Ordering::SeqCst);
                }

                CrawlFeedback::follow_all()
            }) as Pin<Box<dyn Future<Output = CrawlFeedback>>>
        },
    )
    .await
    .unwrap();
}

#[derive(Debug, Default)]
struct TailwindHeuristic {
    has_flex_class: bool,
    has_grid_class: bool,
    has_bg_class: bool,
    has_color_class: bool,
    has_margin_class: bool,
    has_padding_class: bool,
    count_other_attrs: usize,
    count_classes: usize,
}

impl TailwindHeuristic {
    fn tailwind(&self) -> bool {
        self.has_flex_class
            && self.has_grid_class
            && self.has_bg_class
            && self.has_color_class
            && self.has_margin_class
            && self.has_padding_class
        // && self.count_other_attrs <= self.count_classes
    }
}

fn tailwind_heuristic_html(html: &Html) -> TailwindHeuristic {
    let mut heuristic = TailwindHeuristic::default();
    tailwind_heuristic(*html.root_element(), &mut heuristic);
    heuristic
}

fn tailwind_heuristic(node: ego_tree::NodeRef<scraper::Node>, heuristic: &mut TailwindHeuristic) {
    if let Some(element) = ElementRef::wrap(node) {
        for (key, value) in element.value().attrs() {
            if key == "class" {
                heuristic.count_classes += 1;
                for class in value.split_whitespace() {
                    if class == "flex" || class.starts_with("flex-") {
                        heuristic.has_flex_class = true;
                    }
                    if class == "grid" || class.starts_with("grid-") {
                        heuristic.has_grid_class = true;
                    }
                    if class.starts_with("bg-") {
                        heuristic.has_bg_class = true;
                    }
                    if class.starts_with("text-") {
                        heuristic.has_color_class = true;
                    }
                    if class.starts_with("m-")
                        || class.starts_with("mx-")
                        || class.starts_with("my-")
                        || class.starts_with("mt-")
                        || class.starts_with("mb-")
                        || class.starts_with("ml-")
                        || class.starts_with("mr-")
                    {
                        heuristic.has_margin_class = true;
                    }
                    if class.starts_with("p-")
                        || class.starts_with("px-")
                        || class.starts_with("py-")
                        || class.starts_with("pt-")
                        || class.starts_with("pb-")
                        || class.starts_with("pl-")
                        || class.starts_with("pr-")
                    {
                        heuristic.has_padding_class = true;
                    }
                }
            } else {
                heuristic.count_other_attrs += 1;
            }
        }
    }
    // Go through the children
    for child in node.children() {
        tailwind_heuristic(child, heuristic);
    }
}

fn write_element(node: ego_tree::NodeRef<scraper::Node>, to: &mut String) {
    if let Some(element) = ElementRef::wrap(node) {
        let el = element.value();
        let name = el.name();
        let filtered = [
            "script", "noscript", "style", "head", "meta", "link", "svg", "path",
        ];
        if !filtered.contains(&name) && !name.contains("-") {
            to.push_str("<");
            to.push_str(name);
            let mut attrs = el.attrs().into_iter().collect::<Vec<_>>();
            attrs.sort();
            for (key, value) in attrs {
                if key.starts_with("data-")
                    || key.starts_with("aria-")
                    || key.starts_with("role")
                    || key == "id"
                    || key == "style"
                {
                    continue;
                }
                to.push_str(" ");
                to.push_str(key);
                to.push_str("=\"");
                if key == "class" {
                    let mut classes = value.split_whitespace().collect::<Vec<_>>();
                    classes.sort();
                    to.push_str(&classes.join(" "));
                } else if key == "href" || key == "src" || key == "srcset" {
                    to.push_str("url");
                } else {
                    to.push_str(value);
                }
                to.push_str("\"");
            }
            to.push_str(">");
            for child in element.children() {
                write_element(child, to);
            }
            to.push_str("</");
            to.push_str(">");
        }
    }
}
