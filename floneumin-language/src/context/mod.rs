pub mod document;
pub mod page;
pub mod search;

trait Context {
    fn prompt(&self) -> String;
}
