use floneum_plugin::exports::plugins::main::definitions::ValueType;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Edge {
    pub start: usize,
    pub end: usize,
    pub ty: ValueType,
}

impl Edge {
    pub fn new(start: usize, end: usize, ty: ValueType) -> Self {
        Self { start, end, ty }
    }
}
