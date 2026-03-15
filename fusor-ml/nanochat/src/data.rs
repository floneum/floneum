use std::array::from_fn;

use rand::{Rng, rngs::StdRng};

use crate::config::{BATCH_SIZE, BLOCK_SIZE, BOS_TOKEN, CHARSET, EOT_TOKEN, SYSTEM_PROMPT, VOCAB_SIZE};

pub struct Tokenizer;

pub struct ChatDataset {
    examples: Vec<ChatExample>,
}

pub struct ChatExample {
    user: String,
    assistant: String,
    tokens: Vec<u32>,
    assistant_target_start: usize,
}

pub struct Batch {
    pub windows: [[u32; BLOCK_SIZE + 1]; BATCH_SIZE],
    pub mask: [[f32; BLOCK_SIZE]; BATCH_SIZE],
    pub valid_tokens: f32,
}

impl Tokenizer {
    pub fn encode_text(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|ch| self.encode_char(ch))
            .collect()
    }

    pub fn encode_chat_example(&self, user: &str, assistant: &str) -> Vec<u32> {
        let mut tokens = vec![BOS_TOKEN];
        tokens.extend(self.encode_text("system: "));
        tokens.extend(self.encode_text(SYSTEM_PROMPT));
        tokens.extend(self.encode_text("\nuser: "));
        tokens.extend(self.encode_text(user));
        tokens.extend(self.encode_text("\nassistant: "));
        tokens.extend(self.encode_text(assistant));
        tokens.push(EOT_TOKEN);
        tokens
    }

    pub fn encode_chat_prompt(&self, user: &str) -> Vec<u32> {
        let mut tokens = vec![BOS_TOKEN];
        tokens.extend(self.encode_text("system: "));
        tokens.extend(self.encode_text(SYSTEM_PROMPT));
        tokens.extend(self.encode_text("\nuser: "));
        tokens.extend(self.encode_text(user));
        tokens.extend(self.encode_text("\nassistant: "));
        tokens
    }

    pub fn decode_text(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .copied()
            .filter_map(|token| self.decode_token(token))
            .collect()
    }

    pub fn decode_assistant_reply(&self, prompt_tokens: usize, tokens: &[u32]) -> String {
        let generated: Vec<u32> = tokens[prompt_tokens..]
            .iter()
            .copied()
            .take_while(|&token| token != EOT_TOKEN)
            .collect();
        self.decode_text(&generated).trim().to_string()
    }

    fn encode_char(&self, ch: char) -> u32 {
        CHARSET
            .chars()
            .position(|candidate| candidate == ch)
            .unwrap_or_else(|| panic!("character {ch:?} missing from nanochat charset")) as u32
    }

    fn decode_token(&self, token: u32) -> Option<char> {
        CHARSET.chars().nth(token as usize)
    }
}

impl ChatDataset {
    pub fn from_tsv(text: &str, tokenizer: &Tokenizer) -> Self {
        let examples = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(|line| {
                let (user, assistant) = line
                    .split_once('\t')
                    .expect("chat.txt must contain tab-separated user and assistant text");
                let prompt = tokenizer.encode_chat_prompt(user);
                let tokens = tokenizer.encode_chat_example(user, assistant);
                ChatExample {
                    user: user.to_string(),
                    assistant: assistant.to_string(),
                    tokens,
                    // The mask applies to positions whose next token belongs to the assistant reply.
                    assistant_target_start: prompt.len().saturating_sub(1),
                }
            })
            .collect();
        Self { examples }
    }

    pub fn num_docs(&self) -> usize {
        self.examples.len()
    }

    pub fn num_tokens(&self) -> usize {
        self.examples.iter().map(|example| example.tokens.len()).sum()
    }

    pub fn sample_batch(&self, rng: &mut StdRng) -> Batch {
        let sampled: [([u32; BLOCK_SIZE + 1], [f32; BLOCK_SIZE], f32); BATCH_SIZE] =
            from_fn(|_| self.sample_example(rng));

        Batch {
            windows: from_fn(|index| sampled[index].0),
            mask: from_fn(|index| sampled[index].1),
            valid_tokens: sampled.iter().map(|(_, _, valid)| *valid).sum(),
        }
    }

    pub fn max_tokens_per_example(&self) -> usize {
        self.examples
            .iter()
            .map(|example| example.tokens.len())
            .max()
            .unwrap_or(0)
    }

    pub fn examples(&self) -> &[ChatExample] {
        &self.examples
    }

    fn sample_example(&self, rng: &mut StdRng) -> ([u32; BLOCK_SIZE + 1], [f32; BLOCK_SIZE], f32) {
        let example = &self.examples[rng.random_range(0..self.examples.len())];
        let mut window = [EOT_TOKEN; BLOCK_SIZE + 1];
        let mut mask = [0.0; BLOCK_SIZE];

        assert!(
            example.tokens.len() <= BLOCK_SIZE + 1,
            "example length {} exceeds block size {}",
            example.tokens.len(),
            BLOCK_SIZE + 1
        );

        window[..example.tokens.len()].copy_from_slice(&example.tokens);
        let last_valid_input = example.tokens.len().saturating_sub(1);
        if last_valid_input > example.assistant_target_start {
            mask[example.assistant_target_start..last_valid_input].fill(1.0);
        }
        let valid = mask.iter().sum();
        (window, mask, valid)
    }
}

impl ChatExample {
    pub fn user(&self) -> &str {
        &self.user
    }

    pub fn assistant(&self) -> &str {
        &self.assistant
    }
}

pub fn windows_to_inputs(
    windows: &[[u32; BLOCK_SIZE + 1]; BATCH_SIZE],
) -> [[[f32; VOCAB_SIZE]; BLOCK_SIZE]; BATCH_SIZE] {
    let tokens = from_fn(|batch| from_fn(|index| windows[batch][index]));
    one_hot(&tokens)
}

pub fn windows_to_targets(
    windows: &[[u32; BLOCK_SIZE + 1]; BATCH_SIZE],
) -> [[[f32; VOCAB_SIZE]; BLOCK_SIZE]; BATCH_SIZE] {
    let tokens = from_fn(|batch| from_fn(|index| windows[batch][index + 1]));
    one_hot(&tokens)
}

pub fn autoregressive_context(tokens: &[u32]) -> ([u32; BLOCK_SIZE], usize) {
    let mut context = [EOT_TOKEN; BLOCK_SIZE];
    let slice = if tokens.len() > BLOCK_SIZE {
        &tokens[tokens.len() - BLOCK_SIZE..]
    } else {
        tokens
    };
    context[..slice.len()].copy_from_slice(slice);
    (context, slice.len().saturating_sub(1))
}

pub fn one_hot<const B: usize, const T: usize>(tokens: &[[u32; T]; B]) -> [[[f32; VOCAB_SIZE]; T]; B] {
    from_fn(|batch| {
        from_fn(|position| {
            let token = tokens[batch][position] as usize;
            from_fn(|vocab| if vocab == token { 1.0 } else { 0.0 })
        })
    })
}

pub fn position_one_hot() -> [[f32; BLOCK_SIZE]; BLOCK_SIZE] {
    from_fn(|position| from_fn(|column| if column == position { 1.0 } else { 0.0 }))
}
