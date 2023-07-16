pub use crate::exports::plugins::main::definitions::{
    Definition, Definitions, Input, IoDefinition, Output, PrimitiveValue, PrimitiveValueType,
    ValueType,
};
use crate::plugins::main::imports::*;
pub use crate::plugins::main::imports::{get_request, Header};
pub use crate::plugins::main::types::Embedding;
use crate::plugins::main::types::{
    EitherStructure, NumberParameters, SequenceParameters, Structure, ThenStructure, UnsignedRange,
};
pub use crate::plugins::main::types::{EmbeddingDbId, GptNeoXType, LlamaType, ModelType, MptType};
pub use floneum_rust_macro::export_plugin;
use std::ops::RangeInclusive;

#[derive(Debug, Clone, Copy)]
pub struct Structured {
    pub(crate) id: StructureId,
}

impl Structured {
    pub fn literal(text: impl Into<String>) -> Self {
        let inner = Structure::Literal(text.into());

        let id = create_structure(&inner);
        Structured { id }
    }

    pub fn sequence_of(
        item: Structured,
        separator: Structured,
        range: RangeInclusive<u64>,
    ) -> Self {
        let inner = Structure::Sequence(SequenceParameters {
            item: item.id,
            separator: separator.id,
            min_len: *range.start(),
            max_len: *range.end(),
        });
        let id = create_structure(&inner);
        Structured { id }
    }

    pub fn float() -> Self {
        Self::ranged_float(f64::MIN..=f64::MAX)
    }

    pub fn ranged_float(range: RangeInclusive<f64>) -> Self {
        Self::number(range, false)
    }

    pub fn int() -> Self {
        Self::ranged_int(f64::MIN..=f64::MAX)
    }

    pub fn ranged_int(range: RangeInclusive<f64>) -> Self {
        Self::number(range, true)
    }

    pub fn number(range: RangeInclusive<f64>, int: bool) -> Self {
        let inner = Structure::Num(NumberParameters {
            min: *range.start(),
            max: *range.end(),
            integer: int,
        });
        let id = create_structure(&inner);
        Structured { id }
    }

    pub fn str() -> Self {
        Self::ranged_str(0, u64::MAX)
    }

    pub fn ranged_str(min_len: u64, max_len: u64) -> Self {
        let inner = Structure::Str(UnsignedRange {
            min: min_len,
            max: max_len,
        });
        let id = create_structure(&inner);
        Structured { id }
    }

    pub fn boolean() -> Self {
        Self::literal("true").or(Self::literal("false"))
    }

    pub fn null() -> Self {
        Self::literal("null")
    }

    pub fn or_not(self) -> Self {
        self.or(Self::null())
    }

    pub fn or(self, second: Structured) -> Self {
        let inner = Structure::Or(EitherStructure {
            first: self.id,
            second: second.id,
        });
        let id = create_structure(&inner);
        Structured { id }
    }

    pub fn then(self, then: Structured) -> Self {
        let inner = Structure::Then(ThenStructure {
            first: self.id,
            second: then.id,
        });
        let id = create_structure(&inner);
        Structured { id }
    }
}
