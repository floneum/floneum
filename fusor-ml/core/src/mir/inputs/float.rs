use std::fmt::Display;

#[derive(Clone)]
pub(crate) struct FloatInput {
    pub(crate) index: u32,
}

impl Display for FloatInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i_{}", self.index)
    }
}
