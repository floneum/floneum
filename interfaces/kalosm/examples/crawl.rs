use kalosm::language::*;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let count = Arc::new(AtomicUsize::new(0));
    let real_visited = Arc::new(AtomicUsize::new(0));
    Page::crawl(
        Url::parse("https://www.nytimes.com/live/2023/09/21/world/zelensky-russia-ukraine-news")
            .unwrap(),
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

                let Ok(page) = page.article().await else {
                    return CrawlFeedback::follow_none();
                };

                let body = page.body();

                if body.len() < 100 {
                    return CrawlFeedback::follow_none();
                }

                println!("Title: {}", page.title());
                println!("Article:\n{}", body);

                count.fetch_add(1, Ordering::SeqCst);

                CrawlFeedback::follow_all()
            }) as Pin<Box<dyn Future<Output = CrawlFeedback>>>
        },
    )
    .await
    .unwrap();
}
