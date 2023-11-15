use crate::CreateParserState;
use crate::DynTokenizer;
use crate::Parser;
use crate::Tokenizer;
use llm_samplers::prelude::Logit;
use llm_samplers::prelude::Sampler;
use llm_samplers::types::{HasSamplerResources, Logits};
use std::fmt::Debug;

/// A sampler that enforces the given validator. Any tokens that form an invalid sequence will have a probability of 0
pub struct StructuredSampler<V: Parser<Error = E, Output = O, PartialState = PA>, E, O, PA> {
    pub(crate) structure: V,
    pub(crate) current_token_count: usize,
    pub(crate) tokenizer: DynTokenizer,
    pub(crate) sampled: Option<Logit>,
}

impl<V: Parser<Error = E, Output = O, PartialState = PA>, E, O, PA> StructuredSampler<V, E, O, PA> {
    /// Create a new structured sampler that starts validating tokens at the given token count.
    // TODO: improve the current_token_count API
    pub fn new(
        structure: V,
        current_token_count: usize,
        tokenizer: impl Into<DynTokenizer>,
    ) -> Self {
        let tokenizer = tokenizer.into();
        Self {
            structure,
            current_token_count,
            tokenizer,
            sampled: None,
        }
    }
}

impl<V: Parser<Error = E, Output = O, PartialState = PA>, E, O, PA> Debug
    for StructuredSampler<V, E, O, PA>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StructuredSampler").finish()
    }
}

impl<
        V: Parser<Error = E, Output = O, PartialState = PA> + CreateParserState + Send + Sync,
        E,
        O,
        PA,
    > Sampler for StructuredSampler<V, E, O, PA>
{
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources,
        logits: &'a mut Logits,
    ) -> anyhow::Result<&'a mut Logits> {
        let mut valid_tokens = 0;
        let mut best_token: Option<Logit> = None;
        res.with_last_tokens(&mut |previous_tokens| {
            let tokens = &previous_tokens[self.current_token_count..];
            let tokens = match self.tokenizer.decode(tokens) {
                Ok(tokens) => tokens,
                Err(_) => String::new().into(),
            };

            let single_tokens = logits.iter().map(|tid| [tid.token_id]).collect::<Vec<_>>();
            let single_tokens_ref = single_tokens
                .iter()
                .map(|v| v.as_slice())
                .collect::<Vec<_>>();
            let new_tokens = self.tokenizer.decode_batch(&single_tokens_ref).unwrap();
            let mut new_tokens = new_tokens.into_iter();

            for logit in logits.iter_mut() {
                let new_token = new_tokens.next().unwrap();
                if new_token.is_empty() {
                    logit.logit = 0.0;
                    continue;
                }
                let string = tokens.to_string() + &new_token;

                let status = self
                    .structure
                    .parse(&self.structure.create_parser_state(), string.as_bytes());

                match status {
                    Ok(crate::ParseResult::Finished { remaining, .. }) => {
                        if !remaining.is_empty() {
                            logit.logit = 0.0;
                        }
                    }
                    Ok(crate::ParseResult::Incomplete { .. }) => {
                        valid_tokens += 1;
                        if best_token.is_none() || logit.logit > best_token.as_ref().unwrap().logit
                        {
                            best_token = Some(logit.clone());
                        }
                    }
                    Err(_) => {
                        logit.logit = 0.0;
                    }
                }
            }
        })?;
        self.sampled = best_token;
        if valid_tokens == 0 {
            Err(anyhow::anyhow!("No valid tokens"))
        } else {
            Ok(logits)
        }
    }

    fn sampled_token_id(&self) -> Option<u32> {
        self.sampled.as_ref().map(|l| l.token_id)
    }
}
