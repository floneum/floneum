use dioxus_signals::Readable;
use floneum_plugin::plugins::main::types::*;
use serde::{Deserialize, Serialize};

use crate::{application_state, edge::ConnectionType};

#[derive(Serialize, Deserialize)]
pub struct NodeInput {
    pub definition: IoDefinition,
    pub value: Vec<Vec<PrimitiveValue>>,
}

impl NodeInput {
    pub fn new(definition: IoDefinition, value: Vec<Vec<PrimitiveValue>>) -> Self {
        Self { definition, value }
    }

    pub fn set_connection(&mut self, connection: ConnectionType, value: Vec<PrimitiveValue>) {
        match connection {
            ConnectionType::Single => {
                self.value = vec![value];
            }
            ConnectionType::Element(index) => self.set_value(index, value),
        }
    }

    pub fn set_value(&mut self, index: usize, value: Vec<PrimitiveValue>) {
        self.value[index] = value;
    }

    pub fn push_value(&mut self, value: Vec<PrimitiveValue>) {
        if let ValueType::Many(_) = self.definition.ty {
            self.value.push(value);
        }
    }

    pub fn push_default_value(&mut self) -> anyhow::Result<()> {
        if let ValueType::Many(values) = self.definition.ty {
            let application_state = application_state();
            let read = application_state.read();
            let value = values.create(&read.resource_storage)?;
            self.value.push(vec![value]);
        }

        Ok(())
    }

    pub fn pop_value(&mut self) {
        if let ValueType::Many(_) = self.definition.ty {
            self.value.pop();
        }
    }

    pub fn value(&self) -> Vec<PrimitiveValue> {
        self.value.iter().flatten().cloned().collect()
    }
}

#[derive(Serialize, Deserialize)]
pub struct NodeOutput {
    pub definition: IoDefinition,
    pub value: Vec<PrimitiveValue>,
}

impl NodeOutput {
    pub fn as_input(&self) -> Vec<PrimitiveValue> {
        self.value.iter().map(|v| v.borrow()).collect()
    }
}
