use crate::Tokenizer;
use std::borrow::Cow;

impl From<&llm::Tokenizer> for crate::DynTokenizer {
    fn from(tokenizer: &llm::Tokenizer) -> Self {
        Self::new(match tokenizer {
            llm::Tokenizer::Embedded(embedded) => llm::Tokenizer::Embedded(embedded.clone()),
            llm::Tokenizer::HuggingFace(hugging_face) => {
                llm::Tokenizer::HuggingFace(hugging_face.clone())
            }
        })
    }
}

impl Tokenizer for llm::Tokenizer {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        Ok(self
            .tokenize(text, false)?
            .into_iter()
            .map(|token| token.1)
            .collect())
    }

    fn decode(&self, ids: &[u32]) -> anyhow::Result<Cow<'_, str>> {
        let bytes = self.decode(ids.into(), false);
        Ok(String::from_utf8(bytes)?.into())
    }

    fn get_all_tokens(&self) -> anyhow::Result<Cow<'_, [u32]>> {
        anyhow::bail!("Not implemented")
    }
}
