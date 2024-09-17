use std::{collections::HashMap, fmt::Write};

use colored::{Color, Colorize};
use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Merge {
    rank: u32,
    pair: [u32; 2],
    new_token: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MergePriority {
    level: u8,
    rank: u16,
    new_token: u32,
}

impl MergePriority {
    const DEFAULT: Self = Self {
        level: u8::MAX,
        rank: u16::MAX,
        new_token: u32::MAX,
    };
}

impl MergePriority {
    fn new(new_token: u32, rank: u16, level: u8) -> Self {
        MergePriority {
            level,
            rank,
            new_token,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SerializedModel {
    vocab: std::collections::HashMap<String, u32>,
    merges: Vec<String>,
}

fn normalize_token(token: &str) -> String {
    token.replace('Ġ', " ")
}

fn serialize_regex<S>(regex: &Regex, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(regex.as_str())
}

fn deserialize_regex<'de, D>(deserializer: D) -> Result<Regex, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Regex::new(&s).map_err(serde::de::Error::custom)
}

#[derive(Serialize, Deserialize)]
pub struct FastBPETokenizer {
    passes: u8,
    tokens: Vec<Vec<u8>>,
    #[serde(with = "serde_big_array::BigArray")]
    byte_to_token: [u32; 256],
    single_pass_merges: FxHashMap<[u32; 2], MergePriority>,
    #[serde(
        serialize_with = "serialize_regex",
        deserialize_with = "deserialize_regex"
    )]
    regex: Regex,
}

impl FastBPETokenizer {
    pub fn load_from_bytes(bytes: &[u8]) -> Self {
        let json = serde_json::from_slice::<Value>(bytes).unwrap();
        let pretokenizer = json["pre_tokenizer"].clone();
        let sequence = pretokenizer["pretokenizers"][0].clone();
        let pattern = sequence["pattern"]["Regex"].as_str().unwrap();
        let regex = Regex::new(pattern).unwrap();
        let model = json["model"].clone();
        let deserialized = serde_json::from_value::<SerializedModel>(model).unwrap();

        let vocab: HashMap<_, _> = deserialized
            .vocab
            .into_iter()
            .map(|(k, v)| {
                let k = normalize_token(&k);
                (k.as_bytes().to_vec(), v)
            })
            .collect();
        let mut vocab_sorted: Vec<_> = vocab.iter().map(|(k, v)| (k.clone(), *v)).collect();
        vocab_sorted.sort_by_key(|(_, v)| *v);
        let tokens: Vec<_> = vocab_sorted.into_iter().map(|(k, _)| k).collect();

        let merges: Vec<_> = deserialized
            .merges
            .into_iter()
            .enumerate()
            .map(|(rank, merge)| {
                let (first, second) = merge.split_once(' ').unwrap();
                let first = normalize_token(first);
                let second = normalize_token(second);
                let first_bytes = first.as_bytes();
                let second_bytes = second.as_bytes();
                let merged: Vec<u8> = first_bytes
                    .iter()
                    .chain(second_bytes.iter())
                    .copied()
                    .collect();
                let new_token = *vocab.get(&merged).unwrap();
                debug_assert_eq!(merged, tokens[new_token as usize]);
                let first = *vocab.get(first_bytes).unwrap();
                let second = *vocab.get(second_bytes).unwrap();
                Merge {
                    rank: rank as u32,
                    pair: [first, second],
                    new_token,
                }
            })
            .collect();
        let token_to_merge: FxHashMap<_, _> = merges
            .iter()
            .map(|merge| (merge.new_token, *merge))
            .collect();

        let mut current_pass_merges: Vec<Vec<usize>> = Vec::new();
        let mut stack = Vec::new();
        let mut tokens_used_to_create_merge = FxHashSet::default();

        'o: for (merge_idx, merge) in merges.iter().enumerate() {
            tokens_used_to_create_merge.clear();
            stack.clear();
            stack.extend(merge.pair);
            while let Some(token) = stack.pop() {
                if let Some(merge) = token_to_merge.get(&token) {
                    let [first, second] = merge.pair;
                    stack.push(first);
                    stack.push(second);
                    tokens_used_to_create_merge.insert(token);
                }
            }

            let mut index = current_pass_merges.len();
            while index > 0 {
                index -= 1;

                let mut stop_here = false;

                // Check if the new token is a prefix of any existing merge or a postfix of any existing merge. If it is, use the last match after that.
                for other in current_pass_merges[index].iter().copied() {
                    let other_merge = &merges[other];
                    stop_here |= other_merge.pair.contains(&merge.pair[0])
                        || other_merge.pair.contains(&merge.pair[1]);
                    if tokens_used_to_create_merge.contains(&other_merge.new_token) {
                        if index < current_pass_merges.len() - 1 {
                            // If it does conflict, but we fit in at least one previous merge, add the merge to the previous merge
                            current_pass_merges[index + 1].push(merge_idx);
                        } else {
                            // Otherwise, add the merge to the current pass
                            current_pass_merges.push(vec![merge_idx]);
                        }
                        continue 'o;
                    }
                }

                // If the new token would eat a token that is used by this layer, stop here. This is a conflict.
                if stop_here {
                    current_pass_merges[index].push(merge_idx);
                    continue 'o;
                }
            }
            if current_pass_merges.is_empty() {
                // If there are no previous merges, add the merge to the current pass
                current_pass_merges.push(vec![merge_idx]);
            } else {
                // Otherwise, add the merge to the first item
                current_pass_merges[0].push(merge_idx);
            }
        }

        let single_pass_merges: FxHashMap<[u32; 2], MergePriority> = current_pass_merges
            .drain(..)
            .enumerate()
            .flat_map(|(priority, mut v)| {
                v.sort_by_key(|m| merges[*m].rank);
                let merges = &merges;
                v.into_iter().enumerate().map(move |(rank, i)| {
                    let merge = merges[i];
                    (
                        merge.pair,
                        MergePriority::new(
                            merge.new_token,
                            rank.try_into().unwrap(),
                            priority as u8,
                        ),
                    )
                })
            })
            .collect();

        let mut byte_to_token = [u32::MAX; 256];
        for byte in 0..255 {
            if let Some(token) = tokens.iter().position(|v| v == &[byte]) {
                byte_to_token[byte as usize] = token as u32;
            }
        }

        let passes = single_pass_merges
            .values()
            .map(|v| v.level)
            .max()
            .unwrap_or_default();

        Self {
            byte_to_token,
            single_pass_merges,
            tokens,
            passes,
            regex,
        }
    }

    pub fn detokenize<'a>(
        &'a self,
        tokens: impl IntoIterator<Item = u32> + 'a,
    ) -> impl Iterator<Item = &[u8]> + 'a {
        tokens
            .into_iter()
            .map(move |token| self.tokens[token as usize].as_slice())
    }
}

pub struct LayersUsed([usize; 256]);

impl LayersUsed {
    fn new() -> Self {
        Self([0; 256])
    }

    fn reset(&mut self) {
        for layer in self.0.iter_mut() {
            *layer = 0;
        }
    }

    fn set(&mut self, layer: u8, index: usize, len: usize) {
        let index_from_end = len - index;
        let mutable = unsafe { self.0.get_unchecked_mut(layer as usize) };
        *mutable = (*mutable).max(index_from_end);
    }

    fn first_used_index(&self, layer: u8, len: usize) -> usize {
        let index_from_end = unsafe { *self.0.get_unchecked(layer as usize) };
        len.saturating_sub(index_from_end)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TokenData {
    token: u32,
    merge: MergePriority,
}

impl TokenData {
    pub const DEFAULT: Self = Self {
        token: u32::MAX,
        merge: MergePriority::DEFAULT,
    };

    pub fn token(&self) -> u32 {
        self.token
    }
}

// The merge layer queue modifies the bpe buffer in place
// It keeps track of three indexes:
// 1) The index of the bpe buffer that is resolved
// 2) The index of the bpe buffer in the current decreasing subsequence run
// 3) The index of the bpe buffer that is currently being added
pub struct MergeLayerQueue {
    // The index before which all merges in this layer are resolved
    resolved_index: usize,
    // The index that starts the decreasing subsequence run
    decreasing_subsequence_run_start: usize,
    // The index after which nothing has been processed
    current_index: usize,
    // If the last token that was added to [`Self::resolved_index`] was unmodified
    last_unchanged_from_level: bool,
}

impl Default for MergeLayerQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl MergeLayerQueue {
    pub fn new() -> Self {
        Self {
            resolved_index: 0,
            decreasing_subsequence_run_start: 0,
            current_index: 0,
            last_unchanged_from_level: false,
        }
    }

    fn resolve_last_merge(
        &mut self,
        tokens: &mut [TokenData],
        token: u32,
        merges_map: &FxHashMap<[u32; 2], MergePriority>,
        layers_used: &mut LayersUsed,
    ) {
        let tokens_len = tokens.len();
        if let Some(prev_index) = self.resolved_index.checked_sub(1) {
            if let Some(next) = tokens.get_mut(prev_index) {
                if let Some(merge) = merges_map.get(&[next.token, token]) {
                    layers_used.set(merge.level, prev_index, tokens_len);
                    next.merge = *merge;
                }
            }
        }
    }

    fn add_unprocessed_raw_and_calculate_merge(
        &mut self,
        tokens: &mut [TokenData],
        token: TokenData,
        merges_map: &FxHashMap<[u32; 2], MergePriority>,
        layers_used: &mut LayersUsed,
    ) {
        self.resolve_last_merge(tokens, token.token, merges_map, layers_used);

        self.add_unprocessed_raw(tokens, token);
    }

    fn add_unprocessed(
        &mut self,
        tokens: &mut [TokenData],
        token: TokenData,
        merges_map: &FxHashMap<[u32; 2], MergePriority>,
        layers_used: &mut LayersUsed,
    ) {
        if self.last_unchanged_from_level {
            self.add_unprocessed_raw(tokens, token);
        } else {
            self.add_unprocessed_raw_and_calculate_merge(tokens, token, merges_map, layers_used);
        }
    }

    fn add_unprocessed_raw(&mut self, tokens: &mut [TokenData], token: TokenData) {
        unsafe {
            *tokens.get_unchecked_mut(self.resolved_index) = token;
        }
        self.resolved_index += 1;
    }

    pub fn resolve(
        &mut self,
        tokens: &mut Vec<TokenData>,
        input: &str,
        tokenizer: &FastBPETokenizer,
    ) -> usize {
        let mut fill_index = 0;
        let mut layers_used = LayersUsed::new();
        for regex_match in tokenizer.regex.find_iter(input) {
            let mut buffer_processed_end = regex_match.len();
            tokens.resize(fill_index + buffer_processed_end, TokenData::DEFAULT);
            let token = regex_match.as_str();
            layers_used.reset();
            let bytes = token.as_bytes();
            for (i, &b) in bytes.iter().enumerate() {
                let token = unsafe { *tokenizer.byte_to_token.get_unchecked(b as usize) };
                let data = TokenData {
                    token,
                    merge: {
                        let next_index = i + 1;
                        if next_index < bytes.len() {
                            let next = bytes[next_index];
                            match tokenizer
                                .single_pass_merges
                                .get(&[token, tokenizer.byte_to_token[next as usize]])
                            {
                                Some(merge) => {
                                    layers_used.set(merge.level, i, bytes.len());
                                    *merge
                                }
                                None => MergePriority::DEFAULT,
                            }
                        } else {
                            MergePriority::DEFAULT
                        }
                    },
                };
                unsafe {
                    *tokens.get_unchecked_mut(fill_index + i) = data;
                }
            }

            for i in 0..=tokenizer.passes {
                let token_buffer = unsafe {
                    tokens.get_unchecked_mut(fill_index..fill_index + buffer_processed_end)
                };
                if tracing::enabled!(tracing::Level::TRACE) {
                    println!("\n{}\n", format!("level: {i}").color(Color::Yellow).bold());
                    pretty_print_tokens(token_buffer.iter().map(|t| t.token), tokenizer);
                }
                buffer_processed_end = self.resolve_level(
                    token_buffer,
                    &tokenizer.single_pass_merges,
                    &mut layers_used,
                    i,
                    tokenizer,
                );
            }

            fill_index += buffer_processed_end;
        }

        fill_index
    }

    fn pretty_print_info(&self, tokens: &[TokenData], tokenizer: &FastBPETokenizer) {
        print!("> ");
        let mut next_line = String::new();
        for (i, token) in tokens.iter().enumerate() {
            let uncolored = format!("{}-{}", token.merge.level, token.merge.rank);
            let merge_priority = format!(
                "{}-{}",
                token.merge.level.to_string().color(Color::Yellow),
                token.merge.rank.to_string().color(Color::Green)
            );
            let token =
                std::str::from_utf8(tokenizer.tokens[token.token as usize].as_slice()).unwrap();
            write!(&mut next_line, "{}", " ".repeat(token.len())).unwrap();
            write!(&mut next_line, " {} ", merge_priority).unwrap();
            if i < self.resolved_index {
                print!("{}", token.color(Color::Blue))
            } else if i < self.decreasing_subsequence_run_start {
                print!("{}", token.color(Color::White))
            } else if i < self.current_index {
                print!("{}", token.color(Color::Red))
            } else {
                print!("{}", token.color(Color::White))
            }
            print!("{}", " ".repeat(uncolored.len() + 2));
        }
        println!();
        println!("> {next_line}");
    }

    fn resolve_level(
        &mut self,
        tokens: &mut [TokenData],
        merges_map: &FxHashMap<[u32; 2], MergePriority>,
        layers_used: &mut LayersUsed,
        level: u8,
        tokenizer: &FastBPETokenizer,
    ) -> usize {
        self.last_unchanged_from_level = true;

        let start = layers_used.first_used_index(level, tokens.len());
        if start >= tokens.len() {
            return tokens.len();
        }
        self.resolved_index = start;
        self.decreasing_subsequence_run_start = start;
        self.current_index = start;
        let mut last = unsafe { *tokens.get_unchecked(self.current_index) };
        self.current_index += 1;

        while self.current_index < tokens.len() {
            let current_token = unsafe { *tokens.get_unchecked(self.current_index) };
            if tracing::enabled!(tracing::Level::TRACE) {
                tracing::trace!(
                    "tokens [{:?}, {:?}]",
                    std::str::from_utf8(tokenizer.tokens[last.token as usize].as_slice()),
                    std::str::from_utf8(tokenizer.tokens[current_token.token as usize].as_slice())
                );
            }
            let merge_this_and_previous = last.merge;
            // If the level of the merge is not the current level, do not merge yet
            if merge_this_and_previous.level != level {
                // If there is no merge that would use this token, add it directly
                if self.decreasing_subsequence_run_is_empty() {
                    self.add_unprocessed(tokens, last, merges_map, layers_used);
                    self.last_unchanged_from_level = true;
                    self.clear_merge_buffer();
                } else {
                    // Flush the merge buffer and add the current token unprocessed to the buffer
                    self.flush(tokens, merges_map, layers_used);
                }
                self.current_index += 1;
            } else {
                self.current_index += 1;
                // If the next merge is a lower rank than the current merge, do the current merge
                if merge_this_and_previous.level != level
                    || merge_this_and_previous.rank >= current_token.merge.rank
                {
                    // Flush the merge buffer
                    self.flush(tokens, merges_map, layers_used);
                }
                // Otherwise do nothing, the queued merges grow because self.current_index is increased
            }
            last = current_token;
            if tracing::enabled!(tracing::Level::TRACE) {
                self.pretty_print_info(tokens, tokenizer);
            }
        }

        // Just add the last token to the buffer unprocessed
        if self.decreasing_subsequence_run_is_empty() {
            self.add_unprocessed(
                tokens,
                unsafe { *tokens.last().unwrap_unchecked() },
                merges_map,
                layers_used,
            );
            self.last_unchanged_from_level = true;
        } else {
            self.flush(tokens, merges_map, layers_used);
        }

        if tracing::enabled!(tracing::Level::TRACE) {
            self.pretty_print_info(tokens, tokenizer);
        }

        self.resolved_index
    }

    fn decreasing_subsequence_run_is_empty(&self) -> bool {
        self.decreasing_subsequence_run_len() == 0
    }

    fn decreasing_subsequence_run_len(&self) -> usize {
        self.current_index - self.decreasing_subsequence_run_start
    }

    fn clear_merge_buffer(&mut self) {
        self.decreasing_subsequence_run_start = self.current_index;
    }

    fn flush(
        &mut self,
        tokens: &mut [TokenData],
        merges_map: &FxHashMap<[u32; 2], MergePriority>,
        layers_used: &mut LayersUsed,
    ) {
        debug_assert_ne!(self.decreasing_subsequence_run_start, self.current_index);
        let len = self.decreasing_subsequence_run_len();

        let odd_len = len % 2 != 0;
        if odd_len {
            self.add_unprocessed(
                tokens,
                unsafe { *tokens.get_unchecked(self.decreasing_subsequence_run_start) },
                merges_map,
                layers_used,
            );
            self.last_unchanged_from_level = true;
        }
        let mut index = self.decreasing_subsequence_run_start + odd_len as usize;
        while index < self.current_index {
            let token = unsafe { tokens.get_unchecked(index) };
            let token = token.merge.new_token;
            self.add_unprocessed_raw_and_calculate_merge(
                tokens,
                TokenData {
                    token,
                    merge: MergePriority::DEFAULT,
                },
                merges_map,
                layers_used,
            );
            self.last_unchanged_from_level = false;
            index += 2;
        }
        self.clear_merge_buffer();
    }
}

pub fn pretty_print_tokens(resolved: impl Iterator<Item = u32>, tokenizer: &FastBPETokenizer) {
    let colors = [
        Color::Red,
        Color::Green,
        Color::Yellow,
        Color::Blue,
        Color::Magenta,
        Color::Cyan,
    ];
    let mut i = 0;
    for token in tokenizer
        .detokenize(resolved)
        .filter_map(|bytes| std::str::from_utf8(bytes).ok())
    {
        i = (i + 1) % colors.len();
        print!("{}", token.color(colors[i]));
    }
    println!()
}
