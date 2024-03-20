use floneum_plugin::plugins::main::types::*;
use serde::{Deserialize, Serialize};

use crate::edge::ConnectionType;

#[derive(Serialize, Deserialize)]
pub struct NodeInput {
    pub definition: IoDefinition,
    pub value: Vec<Vec<BorrowedPrimitiveValue>>,
}

impl NodeInput {
    pub fn new(definition: IoDefinition, value: Vec<Vec<BorrowedPrimitiveValue>>) -> Self {
        Self { definition, value }
    }

    pub fn set_connection(
        &mut self,
        connection: ConnectionType,
        value: Vec<BorrowedPrimitiveValue>,
    ) {
        match connection {
            ConnectionType::Single => {
                self.value = vec![value];
            }
            ConnectionType::Element(index) => self.set_value(index, value),
        }
    }

    pub fn set_value(&mut self, index: usize, value: Vec<BorrowedPrimitiveValue>) {
        self.value[index] = value;
    }

    pub fn push_value(&mut self, value: Vec<BorrowedPrimitiveValue>) {
        if let ValueType::Many(_) = self.definition.ty {
            self.value.push(value);
        }
    }

    pub fn push_default_value(&mut self) {
        if let ValueType::Many(values) = self.definition.ty {
            let value = values.create();
            self.value.push(vec![value.borrow()]);
        }
    }

    pub fn pop_value(&mut self) {
        if let ValueType::Many(_) = self.definition.ty {
            self.value.pop();
        }
    }

    pub fn value(&self) -> Vec<BorrowedPrimitiveValue> {
        self.value.iter().flatten().cloned().collect()
    }
}

#[derive(Serialize, Deserialize)]
pub struct NodeOutput {
    pub definition: IoDefinition,
    pub value: Vec<PrimitiveValue>,
}

impl NodeOutput {
    pub fn as_input(&self) -> Vec<BorrowedPrimitiveValue> {
        self.value.iter().map(|v| v.borrow()).collect()
    }
}
