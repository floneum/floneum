#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub use futures_util::StreamExt as _;

pub mod language {
    //! Language processing utilities for the Kalosm framework.
    pub use kalosm_language::chat::*;
    pub use kalosm_language::context::*;
    pub use kalosm_language::index::*;
    pub use kalosm_language::kalosm_language_model::{Model as _, ModelExt as _, *};
    pub use kalosm_language::kalosm_llama::{Llama, LlamaBuilder, LlamaSession, LlamaSource};
    pub use kalosm_language::kalosm_sample::*;
    pub use kalosm_language::rbert::{Bert, BertBuilder, BertSource, BertSpace};
    pub use kalosm_language::rmistral::{Mistral, MistralBuilder, MistralSource};
    pub use kalosm_language::rphi::{Phi, PhiBuilder, PhiSource};
    pub use kalosm_language::tool::*;
    pub use kalosm_streams::text_stream::*;
}
pub use kalosm_sound as audio;
pub use kalosm_streams::timed_stream::*;
pub use kalosm_vision as vision;

mod evaluate;
pub use evaluate::*;
