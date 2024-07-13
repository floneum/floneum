use anyhow::Result;
use kalosm_sample::Tokenizer;
use llm_samplers::types::{HasSamplerResources, Logits, Sampler, SamplerError};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::sync::Arc;

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream {
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    /// Creates a new token output stream.
    pub fn new(tokenizer: Arc<dyn Tokenizer + Send + Sync>) -> Self {
        Self {
            tokenizer,
            prev_index: 0,
            current_index: 0,
            tokens: Vec::new(),
        }
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens) {
            Ok(str) => Ok(str.to_string()),
            Err(err) => anyhow::bail!("cannot decode: {err}"),
        }
    }

    /// Samples a token from the logits.
    pub fn sample_token(
        &self,
        sampler: &mut impl Sampler,
        mut logits: Logits,
        stop_on: Option<&str>,
    ) -> anyhow::Result<u32> {
        struct SamplerResources<'a, 'b, R: rand::Rng> {
            rng: &'a mut R,
            previous_tokens: &'b [u32],
        }

        impl<R> std::fmt::Debug for SamplerResources<'_, '_, R>
        where
            R: rand::Rng,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("SamplerResources")
                    .field("previous_tokens", &self.previous_tokens)
                    .finish()
            }
        }

        impl<R> HasSamplerResources for SamplerResources<'_, '_, R>
        where
            R: rand::Rng,
        {
            fn with_rng_mut(
                &mut self,
                fun: &mut dyn FnMut(&mut dyn rand::RngCore),
            ) -> Result<(), SamplerError> {
                fun(self.rng);
                Ok(())
            }

            fn with_last_tokens(&self, fun: &mut dyn FnMut(&[u32])) -> Result<(), SamplerError> {
                fun(self.previous_tokens);
                Ok(())
            }
        }
        let mut rng = rand::thread_rng();
        let tokenizer = self.tokenizer.as_ref();
        let previous_tokens = &self.tokens;

        let mut end_tokens = String::new();
        // grab as many characters as the stop_on string has from the end of the previous tokens
        if let Some(stop_on) = stop_on {
            let required_len = stop_on.len();
            let mut previous_token_iter = previous_tokens.iter().rev();
            while end_tokens.len() < required_len {
                match previous_token_iter.next() {
                    Some(token) => {
                        end_tokens = tokenizer
                            .decode(&[*token])
                            .map_err(anyhow::Error::msg)?
                            .to_string()
                            + &end_tokens;
                    }
                    None => {
                        break;
                    }
                }
            }
        }
        for logit in logits.iter_mut() {
            let tid = logit.token_id;
            if let Some(stop_on) = stop_on {
                let token = tokenizer.decode(&[tid]).unwrap();
                let combined = end_tokens.clone() + &token;
                if combined.contains(stop_on) && !combined.ends_with(stop_on) {
                    // if the token contains a stop_on token, but not the end of the string, set the probability to 0
                    logit.prob = 0.0;
                }
            }
        }
        logits
            .sample_token(
                &mut SamplerResources {
                    previous_tokens,
                    rng: &mut rng,
                },
                sampler,
            )?
            .ok_or_else(|| anyhow::anyhow!("No token sampled"))
    }

    /// Returns the next token.
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            Default::default()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?.to_string()
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_ascii() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    /// Peek the next token.
    pub fn peek_tokens(&self, tokens: Vec<u32>) -> Result<Vec<Option<String>>> {
        let prev_text = if self.tokens.is_empty() {
            Default::default()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?.to_string()
        };
        let prev_text_len = prev_text.len();
        let results = tokens
            .par_iter()
            .map_init(
                || self.tokens[self.prev_index..].to_vec(),
                |tokens, token| {
                    tokens.push(*token);
                    let text = self.decode(tokens).ok()?;
                    tokens.pop();
                    if text.len() > prev_text_len && text.chars().last().unwrap().is_ascii() {
                        let text = text.split_at(prev_text_len);
                        Some(text.1.to_string())
                    } else {
                        None
                    }
                },
            )
            .collect();
        Ok(results)
    }

    /// Get the tokens
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Decode the remaining tokens.
    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            Default::default()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    /// Decode all tokens.
    pub fn decode_all(&self) -> Result<String> {
        self.decode(&self.tokens)
    }

    /// Returns the tokenizer.
    pub fn tokenizer(&self) -> &(dyn Tokenizer + Send + Sync) {
        &*self.tokenizer
    }

    /// Clears the token stream.
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}
