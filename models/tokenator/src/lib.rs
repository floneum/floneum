use std::{
    collections::{HashMap, HashSet},
    fmt::Write,
};

use colored::{Color, Colorize};
use regex::Regex;
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
    key: [u32; 2],
    level: u8,
    rank: u16,
    new_token: u32,
}

impl MergePriority {
    const DEFAULT: MergePriority = MergePriority {
        key: [u32::MAX; 2],
        level: u8::MAX,
        rank: u16::MAX,
        new_token: u32::MAX,
    };

    fn new(key: [u32; 2], new_token: u32, rank: u16, level: u8) -> Self {
        MergePriority {
            key,
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
    levels: u8,
    passes_might_merge_table: MergeTable,
    tokens: Vec<Vec<u8>>,
    #[serde(with = "serde_big_array::BigArray")]
    byte_to_token: [u32; 256],
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
        let token_to_merge: HashMap<_, _> = merges
            .iter()
            .map(|merge| (merge.new_token, *merge))
            .collect();

        let mut current_pass_merges: Vec<Vec<usize>> = Vec::new();
        let mut stack = Vec::new();
        let mut tokens_used_to_create_merge = HashSet::new();

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

        let levels = current_pass_merges.len() as u8;
        let merges = current_pass_merges
            .drain(..)
            .enumerate()
            .flat_map(|(priority, mut v)| {
                v.sort_by_key(|m| merges[*m].rank);
                let merges = &merges;
                v.into_iter().enumerate().map(move |(rank, i)| {
                    let merge = merges[i];
                    (
                        rank.try_into().unwrap(),
                        priority as u8,
                        merge.pair,
                        merge.new_token,
                    )
                })
            });

        let mut byte_to_token = [u32::MAX; 256];
        for byte in 0..255 {
            if let Some(token) = tokens.iter().position(|v| v == &[byte]) {
                byte_to_token[byte as usize] = token as u32;
            }
        }

        Self {
            levels,
            passes_might_merge_table: MergeTable::new(merges),
            byte_to_token,
            tokens,
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

    #[cfg_attr(feature = "never-inline", inline(never))]
    fn resolve_last_merge(
        &mut self,
        tokens: &mut [TokenData],
        token: u32,
        merge_table: &MergeTable,
        layers_used: &mut LayersUsed,
    ) {
        let tokens_len = tokens.len();
        if let Some(prev_index) = self.resolved_index.checked_sub(1) {
            let prev = unsafe { tokens.get_unchecked_mut(prev_index) };
            if let Some(merge) = merge_table.get(prev.token, token) {
                layers_used.set(merge.level, prev_index, tokens_len);
                prev.merge = merge;
            } else {
                prev.merge = MergePriority::DEFAULT;
            }
        }
    }

    #[cfg_attr(feature = "never-inline", inline(never))]
    fn add_unprocessed_raw_and_calculate_merge(
        &mut self,
        tokens: &mut [TokenData],
        token: TokenData,
        merge_table: &MergeTable,
        layers_used: &mut LayersUsed,
    ) {
        self.resolve_last_merge(tokens, token.token, merge_table, layers_used);

        self.add_unprocessed_raw(tokens, token);
    }

    #[cfg_attr(feature = "never-inline", inline(never))]
    fn add_unprocessed(
        &mut self,
        tokens: &mut [TokenData],
        token: TokenData,
        merge_table: &MergeTable,
        layers_used: &mut LayersUsed,
    ) {
        if self.last_unchanged_from_level {
            self.add_unprocessed_raw(tokens, token);
        } else {
            self.add_unprocessed_raw_and_calculate_merge(tokens, token, merge_table, layers_used);
        }
    }

    #[cfg_attr(feature = "never-inline", inline(never))]
    fn add_unprocessed_raw(&mut self, tokens: &mut [TokenData], token: TokenData) {
        unsafe {
            *tokens.get_unchecked_mut(self.resolved_index) = token;
        }
        self.resolved_index += 1;
    }

    #[cfg_attr(feature = "never-inline", inline(never))]
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
                            let next_token = tokenizer.byte_to_token[next as usize];
                            match tokenizer.passes_might_merge_table.get(token, next_token) {
                                Some(merge) => {
                                    layers_used.set(merge.level, i, bytes.len());
                                    merge
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

            for i in 0..tokenizer.levels {
                let token_buffer = unsafe {
                    tokens.get_unchecked_mut(fill_index..fill_index + buffer_processed_end)
                };
                if tracing::enabled!(tracing::Level::TRACE) {
                    println!("\n{}\n", format!("level: {i}").color(Color::Yellow).bold());
                    pretty_print_tokens(token_buffer.iter().map(|t| t.token), tokenizer);
                }
                buffer_processed_end =
                    self.resolve_level(token_buffer, &mut layers_used, i, tokenizer);
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

    #[cfg_attr(feature = "never-inline", inline(never))]
    fn resolve_level(
        &mut self,
        tokens: &mut [TokenData],
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
        let mut rank = u16::MAX;

        while self.current_index < tokens.len() {
            let current_token = unsafe { *tokens.get_unchecked(self.current_index) };
            self.current_index += 1;
            if tracing::enabled!(tracing::Level::TRACE) {
                tracing::trace!("self.current_index: {:?}", self.current_index);
                tracing::trace!("self.resolved_index: {:?}", self.resolved_index);
                tracing::trace!(
                    "token {:?}",
                    std::str::from_utf8(tokenizer.tokens[current_token.token as usize].as_slice())
                );
                self.pretty_print_info(tokens, tokenizer);
            }
            // At this point, we need to add last token either to the buffer or the output
            // We add it to the buffer if it is in a decreasing sequence of rank
            // We add it to the output otherwise
            let merge_this_and_next = current_token.merge;
            let continues_decreasing_sequence =
                merge_this_and_next.level == level && merge_this_and_next.rank < rank;
            if continues_decreasing_sequence {
                rank = merge_this_and_next.rank;
            } else {
                // Otherwise it will be added when we flush the buffer
                tracing::trace!("flushing merge buffer because the new token does not fit into the decreasing sequence");
                self.flush(tokens, &tokenizer.passes_might_merge_table, layers_used);
                rank = u16::MAX;
            }
        }

        if tracing::enabled!(tracing::Level::TRACE) {
            self.pretty_print_info(tokens, tokenizer);
        }

        // Flush the buffer if it is not empty
        if !self.decreasing_subsequence_run_is_empty() {
            tracing::trace!("flushing merge buffer because this is the last token");
            self.flush(tokens, &tokenizer.passes_might_merge_table, layers_used);
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

    #[cfg_attr(feature = "never-inline", inline(never))]
    fn flush(
        &mut self,
        tokens: &mut [TokenData],
        merge_table: &MergeTable,
        layers_used: &mut LayersUsed,
    ) {
        debug_assert_ne!(self.decreasing_subsequence_run_start, self.current_index);
        let len = self.decreasing_subsequence_run_len();

        let odd_len = len % 2 != 0;
        if odd_len {
            tracing::trace!(
                "Length {} is odd, adding the last token to the buffer unprocessed",
                len
            );
            self.add_unprocessed(
                tokens,
                unsafe { *tokens.get_unchecked(self.decreasing_subsequence_run_start) },
                merge_table,
                layers_used,
            );
            self.last_unchanged_from_level = true;
        }
        let mut index = self.decreasing_subsequence_run_start + odd_len as usize;
        while index < self.current_index {
            let token = unsafe { tokens.get_unchecked(index) };
            let token = token.merge.new_token;
            tracing::trace!("Adding token {} to the buffer unprocessed", token);
            self.add_unprocessed_raw_and_calculate_merge(
                tokens,
                TokenData {
                    token,
                    merge: MergePriority::DEFAULT,
                },
                merge_table,
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

const SIZE: u32 = 256;
const _: () = {
    const MAX_TOKENS: u32 = 500_000;
    const MAX_DIV: u32 = MAX_TOKENS / SIZE;
    assert!(MAX_DIV < u16::MAX as u32);
    assert!((MAX_DIV + MAX_DIV * u16::MAX as u32) < u32::MAX);
};

#[derive(Clone, Copy, Serialize, Deserialize)]
struct TableValue {
    ptr: u32,
    end: u32,
}

#[derive(Serialize, Deserialize)]
struct MergeTable {
    mod_bitset: Box<[TableValue]>,
    short_bump: Box<[MergePriority]>,
}

impl MergeTable {
    fn new(iter: impl Iterator<Item = (u16, u8, [u32; 2], u32)>) -> Self {
        let mut merge_nested_table = vec![Vec::new(); (SIZE * SIZE) as usize];
        for (id, rank, pair, new_token) in iter {
            let (index, key) = MergeTable::index(pair[0], pair[1]);
            merge_nested_table[index as usize].push(MergePriority::new(key, new_token, id, rank));
        }

        let mut bump_table = Vec::new();
        let merge_table = merge_nested_table
            .into_iter()
            .map(|table| {
                let ptr = bump_table.len() as u32;
                bump_table.extend(table);
                let end = bump_table.len() as u32;
                TableValue { ptr, end }
            })
            .collect();

        Self {
            mod_bitset: merge_table,
            short_bump: bump_table.into_boxed_slice(),
        }
    }

    #[cfg_attr(feature = "never-inline", inline(never))]
    fn index(input: u32, output: u32) -> (u32, [u32; 2]) {
        let input_mod = input % SIZE;
        let output_mod = output % SIZE;
        (input_mod + output_mod * SIZE, [input, output])
    }

    #[cfg_attr(feature = "never-inline", inline(never))]
    fn get(&self, input: u32, output: u32) -> Option<MergePriority> {
        let (index, key) = Self::index(input, output);

        let value = unsafe { self.mod_bitset.get_unchecked(index as usize) };

        let slice = &self.short_bump[value.ptr as usize..value.end as usize];
        slice.iter().find(|item| item.key == key).copied()
    }
}

#[test]
fn test_empty_merge_table() {
    let table = MergeTable::new(std::iter::empty());
    for x in 0..SIZE {
        for y in 0..SIZE {
            assert!(table.get(x, y).is_none());
        }
    }
}

#[test]
fn test_partial_merge_table() {
    let merges = [
        (0, 0, [0, 0], 0),
        (1, 0, [0, 1], 1),
        (2, 0, [0, 2], 2),
        (3, 0, [0, 3], 3),
        (4, 0, [0, 4], 4),
        (5, 0, [0, 5], 5),
        (6, 0, [0, 6], 6),
        (7, 0, [0, 7], 7),
        (8, 0, [0, 8], 8),
        (9, 0, [0, 9], 9),
    ];

    let table = MergeTable::new(merges.iter().copied());
    for x in 0..SIZE {
        for y in 0..SIZE {
            if let Some((rank, level, _, new_token)) =
                merges.iter().find(|(_, _, pair, _)| pair == &[x, y])
            {
                let result = table.get(x, y).unwrap();
                assert_eq!(result.new_token, *new_token);
                assert_eq!(result.level, *level);
                assert_eq!(result.rank, *rank);
            } else {
                assert!(table.get(x, y).is_none());
            }
        }
    }
}

#[test]
fn test_random_merge_table() {
    use rand::prelude::*;

    let mut merged_tokens = HashSet::new();
    let mut merges: Vec<_> = (0..100_000)
        .map(|_| {
            let mut x = rand::random::<u32>() % 100_000;
            let mut y = rand::random::<u32>() % 100_000;
            while merged_tokens.contains(&[x, y]) {
                x = rand::random::<u32>() % 100_000;
                y = rand::random::<u32>() % 100_000;
            }
            merged_tokens.insert([x, y]);
            let rank = rand::random::<u16>() % 100;
            let level = rand::random::<u8>() % 10;
            let new_token = rand::random::<u32>();
            (rank, level, [x, y], new_token)
        })
        .collect();

    merges.shuffle(&mut rand::thread_rng());
    let map = merges
        .iter()
        .map(|(rank, level, pair, new_token)| {
            (pair, MergePriority::new(*pair, *new_token, *rank, *level))
        })
        .collect::<HashMap<_, _>>();

    let table = MergeTable::new(merges.iter().copied());
    for x in 0..SIZE {
        for y in 0..SIZE {
            if let Some(merge_priority) = map.get(&[x, y]) {
                let result = table.get(x, y).unwrap();
                assert_eq!(result, *merge_priority);
            } else {
                assert!(table.get(x, y).is_none());
            }
        }
    }
}