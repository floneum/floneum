pub mod document;
pub mod page;
pub mod rss;
pub mod search;

pub use url::Url;

trait Context {
    fn description(&self) -> String;
    fn prompt(&self) -> String;
}
