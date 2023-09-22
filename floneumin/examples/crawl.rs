use floneumin_language::context::page::Page;
use floneumin_language::context::Url;
use std::cell::Cell;
use std::rc::Rc;

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

    let count = Rc::new(Cell::new(0));
    nyt.crawl(
        move |page| {
            let count = count.clone();
            Box::pin(async move {
                println!("Page: {}", page.url());
                if count.get() > 15 {
                    return false;
                }

                println!("Title: {}", page.title().await.unwrap());
                let page = page.article().await.unwrap();
                let body = page.body();
                println!("Article:\n{}", body);

                if body.len() < 100 {
                    return false;
                }

                count.set(count.get() + 1);

                true
            })
        },
        false,
        false,
    )
    .await
    .unwrap();
}
