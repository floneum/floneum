use floneum_plugin::plugins::main::types::*;
use serde::{Deserialize, Serialize};

use crate::edge::ConnectionType;

#[derive(Serialize, Deserialize)]
pub struct NodeInput {
    pub definition: IoDefinition,
    pub value: Vec<BorrowedPrimitiveValue>,
}

impl NodeInput {
    pub fn new(definition: IoDefinition, value: Vec<BorrowedPrimitiveValue>) -> Self {
        Self { definition, value }
    }

    pub fn set_connection(&mut self, connection: ConnectionType, value: BorrowedPrimitiveValue) {
        match connection {
            ConnectionType::Single => {
                self.value = vec![value];
            }
            ConnectionType::Element(index) => self.set_value(index, value),
        }
    }

    pub fn set_value(&mut self, index: usize, value: Input) {
        self.value[index] = value;
    }

    pub fn push_value(&mut self, value: Vec<BorrowedPrimitiveValue>) {
        if let ValueType::Many(_) = self.definition.ty {
            self.value.push(value);
        }
    }

    pub fn push_default_value(&mut self) {
        if let ValueType::Many(values) = self.definition.ty {
            let value = Input::Single(values.create());
            self.value.push(value);
        }
    }

    pub fn pop_value(&mut self) {
        if let ValueType::Many(_) = self.definition.ty {
            self.value.pop();
        }
    }

    pub fn value(&self) -> Vec<BorrowedPrimitiveValue> {
        self.value
    }
}

#[derive(Serialize, Deserialize)]
pub struct NodeOutput {
    pub definition: IoDefinition,
    pub value: Vec<PrimitiveValue>,
}

impl NodeOutput {
    pub fn as_input(&self) -> Option<Vec<BorrowedPrimitiveValue>> {
        self.value.iter().map(|v| v.borrow()).collect()
    }
}
