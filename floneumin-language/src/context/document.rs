#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    title: String,
    body: String,
}

impl Document {
    pub fn new(title: String, body: String) -> Self {
        Self { title, body }
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub fn body(&self) -> &str {
        &self.body
    }
}
