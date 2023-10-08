
use floneum_plugin::plugins::main::types::*;
use serde::{Deserialize, Serialize};

use crate::edge::ConnectionType;

#[derive(Serialize, Deserialize)]
pub struct NodeInput {
    pub definition: IoDefinition,
    pub value: Vec<Input>,
}

impl NodeInput {
    pub fn new(definition: IoDefinition, value: Input) -> Self {
        Self {
            definition,
            value: vec![value],
        }
    }

    pub fn set_connection(&mut self, connection: ConnectionType, value: Input) {
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

    pub fn push_value(&mut self, value: Input) {
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

    pub fn value(&self) -> Input {
        match self.definition.ty {
            ValueType::Single(_) => self.value[0].clone(),
            ValueType::Many(_) => {
                let mut return_values: Vec<PrimitiveValue> = Vec::new();

                for value in &self.value {
                    match value {
                        Input::Single(value) => {
                            return_values.push(value.clone());
                        }
                        Input::Many(values) => {
                            for value in values {
                                return_values.push(value.clone());
                            }
                        }
                    }
                }

                Input::Many(return_values)
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct NodeOutput {
    pub definition: IoDefinition,
    pub value: Output,
}

impl NodeOutput {
    pub fn as_input(&self) -> Option<Input> {
        match &self.value {
            Output::Single(value) => Some(Input::Single(value.clone())),
            Output::Many(values) => Some(Input::Many(values.clone())),
            Output::Halt => None,
        }
    }
}
