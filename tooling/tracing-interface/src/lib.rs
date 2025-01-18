use std::panic::Location;

use kalosm_sample::SchemaType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    CreateTask(CreateTaskMessage),
    AddToken(AddTokenMessage),
    EndTask(EndTaskMessage),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddTokenMessage {
    id: u64,
    token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTaskMessage {
    id: u64,
    caller: Caller,
    schema: Option<SchemaType>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndTaskMessage {
    id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Caller {
    line: u32,
    column: u32,
    file: String,
}

impl<'a> From<&'a Location<'a>> for Caller {
    fn from(location: &'a Location<'a>) -> Self {
        Self {
            line: location.line(),
            column: location.column(),
            file: location.file().to_string(),
        }
    }
}
