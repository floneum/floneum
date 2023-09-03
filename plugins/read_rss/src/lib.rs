use floneum_rust::*;
use rss::Channel;
use url::Url;

#[export_plugin]
/// Reads a rss stream from a url
pub fn read_rss_stream(
    /// The url to read the rss stream from
    url: String,
    /// The number of items to read
    items: i64,
) -> String {
    let xml = get_request(&url, &[]);
    let channel = match Channel::read_from(xml.as_bytes()) {
        Ok(channel) => channel,
        Err(err) => {
            println!("{}", xml);
            println!("{}", err);
            return String::new();
        }
    };
    channel
        .items()
        .iter()
        .take(items as usize)
        .filter_map(|item| {
            let mut message = String::new();
            if let Some(title) = item.title() {
                message.push_str(&format!("### {}\n", title));
            }
            let (source_url, content) = if let Some(content) = item.content() {
                (None, content.to_string())
            } else if let Some(source_url) = item.link() {
                (Some(source_url), get_request(source_url, &[]))
            } else {
                (None, String::new())
            };

            let url = Url::parse(match source_url {
                Some(url) => url,
                None => &url,
            })
            .unwrap();

            match readability::extractor::extract(&mut std::io::Cursor::new(&content), &url) {
                Ok(article) => {
                    message.push_str(&format!("{}\n\n\n", article.text));
                }
                Err(_) => {
                    return None;
                }
            }

            Some(message)
        })
        .collect::<String>()
}
