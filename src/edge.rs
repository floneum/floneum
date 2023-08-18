use floneum_plugin::exports::plugins::main::definitions::ValueType;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Edge {
    pub start: usize,
    pub end: Connection,
    pub ty: ValueType,
}

impl Edge {
    pub fn new(start: usize, end: Connection, ty: ValueType) -> Self {
        Self { start, end, ty }
    }
}

#[derive(PartialEq, Clone, Copy, Serialize, Deserialize)]
pub struct Connection {
    pub index: usize,
    pub ty: ConnectionType,
}

#[derive(PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum ConnectionType {
    Single,
    Element(usize),
}
