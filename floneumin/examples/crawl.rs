use floneumin_language::context::page::Page;
use floneumin_language::context::Url;

#[tokio::main]
async fn main() {
    let nyt = Page::new(
        Url::parse("https://www.nytimes.com/live/2023/09/21/world/zelensky-russia-ukraine-news")
            .unwrap(),
        false,
        false,
    )
    .await
    .unwrap();

    let mut count = 0;
    nyt.crawl(
        move |page| {
            if count > 5 {
                return false;
            }

            println!("Page: {}", page.url());
            println!("Title: {}", page.title().unwrap());
            let page = page.article().unwrap();
            let body = page.body();
            println!("Article:\n{}", body);

            if body.len() < 100 {
                return false;
            }

            count += 1;

            true
        },
        false,
        false,
    )
    .await
    .unwrap();
}
