pub mod document;
pub mod search;

trait Context {
    fn prompt(&self) -> String;
}
