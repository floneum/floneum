use crate::LocalSubscription;
use floneum_plugin::exports::plugins::main::definitions::Input;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Edge {
    pub start: usize,
    pub end: usize,
    // pub type_bounds: Option<ValueType>,
    pub value: Option<LocalSubscription<Input>>,
}

impl Edge {
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            // type_bounds: None,
            value: None,
        }
    }
}
