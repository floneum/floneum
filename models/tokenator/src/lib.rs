use std::{
    collections::{HashMap, HashSet},
    fmt::Write,
    num::NonZeroU8,
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
    token_size: Vec<u8>,
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

        let mut token_size = vec![None; tokens.len()];
        fn find_token_size(
            token: u32,
            token_to_merge: &HashMap<u32, Merge>,
            token_size: &mut Vec<Option<u8>>,
        ) {
            if token_size[token as usize].is_some() {
                return;
            }
            match token_to_merge.get(&token) {
                Some(merge) => {
                    let [left, right] = merge.pair;
                    find_token_size(left, token_to_merge, token_size);
                    let left_size = token_size[left as usize].unwrap();
                    find_token_size(right, token_to_merge, token_size);
                    let right_size = token_size[right as usize].unwrap();
                    let size = left_size + right_size;
                    token_size[token as usize] = Some(size);
                }
                None => {
                    token_size[token as usize] = Some(1);
                }
            }
        }
        for i in 0..tokens.len() {
            find_token_size(i as u32, &token_to_merge, &mut token_size);
        }

        let token_size = token_size
            .into_iter()
            .map(|v| v.unwrap())
            .collect::<Vec<_>>();

        Self {
            levels,
            passes_might_merge_table: MergeTable::new(merges),
            byte_to_token,
            tokens,
            token_size,
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

#[derive(Clone, Copy, Debug)]
pub struct TokenData {
    token: u32,
    size: NonZeroU8,
    merge: MergePriority,
}

impl TokenData {
    pub const DEFAULT: Self = Self {
        token: u32::MAX,
        size: NonZeroU8::MIN,
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
#[derive(Debug)]
pub struct MergeLayerQueue {
    /// The length of the sparse decreasing subsequence run
    decreasing_subsequence_run_len: usize,
    /// The index after which nothing has been processed
    current_index: usize,
    /// The last resolved token index. We need to keep track of this because the last token may be merged with the next token
    last_token_index: Option<usize>,
    rank: u16,
}

impl Default for MergeLayerQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl MergeLayerQueue {
    pub fn new() -> Self {
        Self {
            decreasing_subsequence_run_len: 0,
            current_index: 0,
            last_token_index: None,
            rank: u16::MAX,
        }
    }

    #[cfg_attr(feature = "never-inline", inline(never))]
    fn add_unprocessed_raw(
        &mut self,
        tokens: &mut [TokenData],
        token: TokenData,
        resolved_index: usize,
    ) {
        unsafe {
            *tokens.get_unchecked_mut(resolved_index) = token;
        }
    }

    #[cfg_attr(feature = "never-inline", inline(never))]
    pub fn resolve(
        tokens: &mut Vec<TokenData>,
        input: &str,
        tokenizer: &FastBPETokenizer,
        levels: &mut Vec<Self>,
    ) -> usize {
        let mut fill_index = 0;
        levels.resize_with(tokenizer.levels as usize, Self::new);
        levels.fill_with(Self::new);
        for regex_match in tokenizer.regex.find_iter(input) {
            let start = fill_index;
            let end = fill_index + regex_match.len();

            tokens.resize(end, TokenData::DEFAULT);

            for level in levels.iter_mut() {
                level.reset();
            }
            let token_buffer = unsafe { tokens.get_unchecked_mut(fill_index..) };
            let bytes = regex_match.as_str().as_bytes();

            #[cfg_attr(feature = "never-inline", inline(never))]
            fn add_chunk_of_bytes<const LAST: bool>(
                start: usize,
                resolved: usize,
                levels: &mut [MergeLayerQueue],
                tokenizer: &FastBPETokenizer,
                bytes: &[u8],
                token_buffer: &mut [TokenData],
                skip_list: &mut [u64; 256 / 64],
            ) {
                let mut prev_level_resolved = resolved;
                fn add_single_byte<const LAST: bool>(
                    index: usize,
                    tokenizer: &FastBPETokenizer,
                    bytes: &[u8],
                    token_buffer: &mut [TokenData],
                    skip_list: &mut [u64; 256 / 64],
                ) {
                    let b = unsafe { *bytes.get_unchecked(index) };
                    let out = unsafe { token_buffer.get_unchecked_mut(index) };
                    let token = unsafe { *tokenizer.byte_to_token.get_unchecked(b as usize) };
                    out.token = token;
                    out.size = NonZeroU8::MIN;

                    let next_index = index + 1;
                    let next = if LAST {
                        bytes.get(next_index).copied()
                    } else {
                        let next = unsafe { *bytes.get_unchecked(next_index) };
                        Some(next)
                    };
                    if let Some(next) = next {
                        let next_token =
                            unsafe { *tokenizer.byte_to_token.get_unchecked(next as usize) };
                        tokenizer.passes_might_merge_table.get(
                            token,
                            next_token,
                            &mut out.merge,
                            skip_list,
                        );
                    } else {
                        out.merge.level = u8::MAX;
                    }
                }
                for i in start..resolved - 1 {
                    add_single_byte::<false>(i, tokenizer, bytes, token_buffer, skip_list);
                }
                add_single_byte::<true>(resolved - 1, tokenizer, bytes, token_buffer, skip_list);
                let next_token = unsafe { token_buffer.get_unchecked(start).token };
                // after adding a new byte, we need to update the merge for the previous token
                // first find the previous token
                let mut prev_token_pos = start;
                let mut prev_token;
                while prev_token_pos > 0 {
                    prev_token_pos -= 1;
                    prev_token = unsafe { token_buffer.get_unchecked_mut(prev_token_pos) };
                    if prev_token.token != u32::MAX {
                        tokenizer.passes_might_merge_table.get(
                            prev_token.token,
                            next_token,
                            &mut prev_token.merge,
                            skip_list,
                        );
                        break;
                    }
                }
                for (i, level) in levels.iter_mut().enumerate() {
                    tracing::trace!("working on level: {i}");
                    tracing::trace!("working up to index: {prev_level_resolved}");
                    debug_assert!(
                        prev_level_resolved <= token_buffer.len(),
                        "{prev_level_resolved} > {}",
                        token_buffer.len()
                    );

                    let mut unchanged = true;
                    if get_bitset(skip_list, i as u8) && prev_level_resolved > 0 {
                        let skip_to = prev_level_resolved;
                        tracing::trace!("skipping to {skip_to} for level {i}");
                        level.skip_to(
                            token_buffer,
                            tokenizer,
                            skip_list,
                            skip_to,
                            #[cfg(debug_assertions)]
                            {
                                i as u8
                            },
                        );
                        unchanged = false;
                    } else {
                        while level.current_index < prev_level_resolved {
                            unchanged &=
                                level.process_token(token_buffer, i as u8, tokenizer, skip_list);
                        }
                    }

                    if LAST {
                        level.finish(
                            token_buffer,
                            tokenizer,
                            skip_list,
                            #[cfg(debug_assertions)]
                            {
                                i as u8
                            },
                        );
                    }
                    if unchanged && !LAST {
                        tracing::trace!("update stopped at level: {i}");
                        break;
                    }
                    prev_level_resolved = level.last_token_index.unwrap_or(0);
                }
            }

            const JUMP_BY: usize = 4;
            let mut last = 0;
            let mut skip_list = [0u64; 256 / 64];
            for resolved in (JUMP_BY..regex_match.len()).step_by(JUMP_BY) {
                add_chunk_of_bytes::<false>(
                    last,
                    resolved,
                    levels,
                    tokenizer,
                    bytes,
                    token_buffer,
                    &mut skip_list,
                );
                last = resolved;
            }
            {
                let resolved = regex_match.len();
                add_chunk_of_bytes::<true>(
                    last,
                    resolved,
                    levels,
                    tokenizer,
                    bytes,
                    token_buffer,
                    &mut skip_list,
                );
            }

            // Compact the resolved tokens
            let mut index = start;
            while index < end {
                let token = unsafe { tokens.get_unchecked(index) };
                index += token.size.get() as usize;
                unsafe {
                    *tokens.get_unchecked_mut(fill_index) = *token;
                }
                fill_index += 1;
            }
        }

        fill_index
    }

    fn pretty_print_info(&self, tokens: &[TokenData], tokenizer: &FastBPETokenizer) {
        print!("> ");
        let mut next_line = String::new();
        let mut dense_index = 0;
        for (i, token) in tokens.iter().enumerate() {
            if token.token == u32::MAX {
                print!(" █ ");
                next_line.push_str("   ");
                continue;
            }
            let in_dense = i == dense_index;
            if in_dense {
                dense_index += tokenizer.token_size[token.token as usize] as usize;
            }
            let uncolored = format!(
                "{}-{}-{}",
                token.merge.level, token.merge.rank, token.merge.new_token
            );
            let merge_priority = format!(
                "{}-{}-{}",
                token.merge.level.to_string().color(Color::Yellow),
                token.merge.rank.to_string().color(Color::Green),
                token.merge.new_token.to_string().color(Color::Red),
            );
            let token =
                std::str::from_utf8(tokenizer.tokens[token.token as usize].as_slice()).unwrap();
            write!(&mut next_line, "{}", " ".repeat(token.len())).unwrap();
            write!(&mut next_line, " {} ", merge_priority).unwrap();
            let resolved_index = self
                .last_token_index
                .map(|idx| idx + unsafe { tokens.get_unchecked(idx) }.size.get() as usize)
                .unwrap_or(0);
            if !in_dense {
                print!("{}", token.color(Color::White))
            } else if i < resolved_index {
                print!("{}", token.color(Color::Blue))
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

    fn reset(&mut self) {
        self.decreasing_subsequence_run_len = 0;
        self.current_index = 0;
        self.last_token_index = None;
        self.rank = u16::MAX;
    }

    fn skip_to(
        &mut self,
        tokens: &mut [TokenData],
        tokenizer: &FastBPETokenizer,
        skip_list: &mut [u64; 256 / 64],
        index: usize,
        #[cfg(debug_assertions)] level: u8,
    ) {
        if !self.decreasing_subsequence_run_is_empty() {
            self.flush(
                tokens,
                &tokenizer.passes_might_merge_table,
                tokenizer,
                skip_list,
                #[cfg(debug_assertions)]
                level,
            );
        }
        self.decreasing_subsequence_run_len = 0;
        self.current_index = index;
        self.rank = u16::MAX;
        self.last_token_index = Some({
            let mut start = index - 1;
            while start > 0 && unsafe { tokens.get_unchecked(start).token == u32::MAX } {
                start -= 1;
            }
            start
        });
    }

    #[cfg_attr(feature = "never-inline", inline(never))]
    fn process_token(
        &mut self,
        tokens: &mut [TokenData],
        level: u8,
        tokenizer: &FastBPETokenizer,
        skip_list: &mut [u64; 256 / 64],
    ) -> bool {
        let current_token = unsafe { tokens.get_unchecked(self.current_index) };
        self.current_index += current_token.size.get() as usize;
        self.decreasing_subsequence_run_len += 1;
        #[cfg(debug_assertions)]
        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!("self.current_index: {:?}", self.current_index);
            tracing::trace!("self.last_token_index: {:?}", self.last_token_index);
            self.pretty_print_info(tokens, tokenizer);
        }
        // At this point, we need to add last token either to the buffer or the output
        // We add it to the buffer if it is in a decreasing sequence of rank
        // We add it to the output otherwise
        let merge_this_and_next = current_token.merge;
        let continues_decreasing_sequence =
            merge_this_and_next.level == level && merge_this_and_next.rank < self.rank;
        if continues_decreasing_sequence {
            self.rank = merge_this_and_next.rank;
        } else {
            // Otherwise it will be added when we flush the buffer
            tracing::trace!("flushing merge buffer because the new token does not fit into the decreasing sequence");
            self.flush(
                tokens,
                &tokenizer.passes_might_merge_table,
                tokenizer,
                skip_list,
                #[cfg(debug_assertions)]
                level,
            );
        }

        #[cfg(debug_assertions)]
        if tracing::enabled!(tracing::Level::TRACE) {
            self.pretty_print_info(tokens, tokenizer);
        }

        continues_decreasing_sequence
    }

    fn finish(
        &mut self,
        tokens: &mut [TokenData],
        tokenizer: &FastBPETokenizer,
        skip_list: &mut [u64; 256 / 64],
        #[cfg(debug_assertions)] level: u8,
    ) {
        // Flush the buffer if it is not empty
        if !self.decreasing_subsequence_run_is_empty() {
            tracing::trace!("flushing merge buffer because this is the last token");
            self.flush(
                tokens,
                &tokenizer.passes_might_merge_table,
                tokenizer,
                skip_list,
                #[cfg(debug_assertions)]
                level,
            );
        }

        if tracing::enabled!(tracing::Level::TRACE) {
            self.pretty_print_info(tokens, tokenizer);
        }
        let resolved_index = self
            .last_token_index
            .map(|idx| idx + unsafe { tokens.get_unchecked(idx) }.size.get() as usize)
            .unwrap_or(0);
        self.last_token_index = Some(resolved_index);
    }

    fn decreasing_subsequence_run_is_empty(&self) -> bool {
        self.decreasing_subsequence_run_len() == 0
    }

    fn decreasing_subsequence_run_len(&self) -> usize {
        self.decreasing_subsequence_run_len
    }

    fn clear_merge_buffer(&mut self) {
        self.decreasing_subsequence_run_len = 0;
    }

    #[cfg_attr(feature = "never-inline", inline(never))]
    fn flush(
        &mut self,
        tokens: &mut [TokenData],
        merge_table: &MergeTable,
        tokenizer: &FastBPETokenizer,
        skip_list: &mut [u64; 256 / 64],
        #[cfg(debug_assertions)] level: u8,
    ) {
        let len = self.decreasing_subsequence_run_len();
        tracing::trace!("flushing {} tokens", len);

        let odd_len = len % 2 != 0;
        let decreasing_subsequence_run_start = self
            .last_token_index
            .map(|idx| idx + unsafe { tokens.get_unchecked(idx) }.size.get() as usize)
            .unwrap_or(0);
        let mut start = decreasing_subsequence_run_start;
        if odd_len {
            tracing::trace!(
                "Length {} is odd, adding the first token to the buffer unprocessed",
                len
            );
            let size = unsafe { *tokens.get_unchecked(decreasing_subsequence_run_start) }
                .size
                .get() as usize;
            self.last_token_index = Some(start);
            start += size;
        }
        let mut index = start;
        while index < self.current_index {
            let token = unsafe { tokens.get_unchecked(index) };
            debug_assert!(token.merge.level != u8::MAX, "{:?}", token);
            let new_token = token.merge.new_token;
            let size = unsafe { *tokenizer.token_size.get_unchecked(new_token as usize) };
            let new_token_data = TokenData {
                token: new_token,
                size: unsafe { NonZeroU8::new_unchecked(size) },
                merge: MergePriority::DEFAULT,
            };
            // The size of the new token should be the sum of the other tokens
            #[cfg(debug_assertions)]
            {
                let left = token.token;
                let left_size = tokenizer.token_size[left as usize] as usize;
                let right = unsafe { tokens.get_unchecked(index + left_size) };
                let right_size = tokenizer.token_size[right.token as usize] as usize;
                assert_eq!(
                    left_size + right_size,
                    size as usize,
                    "{token:?} and {right:?} don't merge into {new_token_data:?}"
                );
            }
            // The merge should be from this level
            #[cfg(debug_assertions)]
            {
                assert_eq!(token.merge.level, level);
            }
            let right_token =
                unsafe { tokens.get_unchecked_mut(index + token.size.get() as usize) };
            right_token.token = u32::MAX;
            // Fix the merge of the last token
            if let Some(last) = self.last_token_index {
                let last_token = unsafe { tokens.get_unchecked_mut(last) };
                merge_table.get(
                    last_token.token,
                    new_token,
                    &mut last_token.merge,
                    skip_list,
                );
            }
            let resolved = index;
            self.last_token_index = Some(index);
            index += size as usize;
            self.add_unprocessed_raw(tokens, new_token_data, resolved);
        }
        // Fix the merge of the last token
        if len != 1 {
            if let Some(last) = self.last_token_index {
                if let Some(next) = tokens.get(index) {
                    let next = next.token;
                    let last_token = unsafe { tokens.get_unchecked_mut(last) };
                    merge_table.get(last_token.token, next, &mut last_token.merge, skip_list);
                }
            }
        }

        self.clear_merge_buffer();
        self.rank = u16::MAX;
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
    fn get(
        &self,
        input: u32,
        output: u32,
        out: &mut MergePriority,
        skip_list: &mut [u64; 256 / 64],
    ) {
        let (index, key) = Self::index(input, output);

        let value = unsafe { self.mod_bitset.get_unchecked(index as usize) };

        let slice = &self.short_bump[value.ptr as usize..value.end as usize];
        if let Some(found) = slice.iter().find(|item| item.key == key) {
            set_bitset(skip_list, found.level);
            *out = *found;
        } else {
            out.level = u8::MAX;
        }
    }
}

fn set_bitset(skip_list: &mut [u64; 256 / 64], level: u8) {
    let skip_value = &mut skip_list[level as usize / 64];
    *skip_value |= 1u64 << (level as usize % 64);
}

fn get_bitset(skip_list: &[u64; 256 / 64], level: u8) -> bool {
    let skip_value = &skip_list[level as usize / 64];
    (*skip_value & (1u64 << (level as usize % 64))) == 0
}

#[test]
fn test_empty_merge_table() {
    let table = MergeTable::new(std::iter::empty());
    for x in 0..SIZE {
        for y in 0..SIZE {
            let mut out = MergePriority::DEFAULT;
            table.get(x, y, &mut out, &mut [0; 4]);
            assert_eq!(out.level, u8::MAX);
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
                let mut result = MergePriority::DEFAULT;
                table.get(x, y, &mut result, &mut [0; 4]);
                assert_eq!(result.new_token, *new_token);
                assert_eq!(result.level, *level);
                assert_eq!(result.rank, *rank);
            } else {
                let mut out = MergePriority::DEFAULT;
                table.get(x, y, &mut out, &mut [0; 4]);
                assert_eq!(out.level, u8::MAX);
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
    let mut skip_list = [0; 4];
    for x in 0..SIZE {
        for y in 0..SIZE {
            if let Some(merge_priority) = map.get(&[x, y]) {
                let mut result = MergePriority::DEFAULT;
                table.get(x, y, &mut result, &mut skip_list);
                assert_eq!(result, *merge_priority);
            } else {
                let mut out = MergePriority::DEFAULT;
                table.get(x, y, &mut out, &mut skip_list);
                assert_eq!(out.level, u8::MAX);
            }
        }
    }
}
