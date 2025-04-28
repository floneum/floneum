use std::fmt::Display;

#[derive(Clone)]
pub(crate) struct IntegerInput {
    pub(crate) index: u32,
}

impl Display for IntegerInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i_{}", self.index)
    }
}
