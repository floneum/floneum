use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Edge {
    pub start: usize,
    pub end: Connection,
}

impl Edge {
    pub fn new(start: usize, end: Connection,) -> Self {
        Self { start, end,}
    }
}

#[derive(PartialEq, Clone, Copy, Serialize, Deserialize, Debug)]
pub struct Connection {
    pub index: usize,
    pub ty: ConnectionType,
}

#[derive(PartialEq, Clone, Copy, Serialize, Deserialize, Debug)]
pub enum ConnectionType {
    Single,
    Element(usize),
}
