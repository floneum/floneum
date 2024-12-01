use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    num::NonZeroU8,
    simd::{num::SimdUint, Simd},
    vec,
};

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tokenize;

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

        // panic!("max current_pass_merges: {:?}", current_pass_merges.iter().map(|v| v.len()).collect::<Vec<_>>());

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
    ) -> impl Iterator<Item = &'a [u8]> + 'a {
        tokens
            .into_iter()
            .map(move |token| self.tokens[token as usize].as_slice())
    }

    pub fn tokenize<'a>(
        &'a self,
        text: &str,
        token_buffer: &mut Vec<u32>,
        level_buffer: &mut Vec<u8>,
        merge_buffer: &mut Vec<u32>,
        merge_priority_buffer: &mut Vec<u16>,
    ) {
        const MASK_SIZE: usize = 16;
        let max: Simd<u32, 32> = Simd::splat(u32::MAX);
        for bytes in text.as_bytes().chunks(16) {
            let bytes = Simd::load_or_default(bytes);
            let bytes_idx = bytes.cast();
            let encoded = Simd::gather_or(&self.byte_to_token, bytes_idx, max);
            token_buffer.extend_from_slice(encoded.as_array());
        }
        for arr in token_buffer.windows(2) {
            let &[first, second] = arr else {
                unreachable!()
            };
            if let Some(merge) = self.passes_might_merge_table.get(first, second) {
                level_buffer.push(merge.level);
                merge_buffer.push(merge.new_token);
                merge_priority_buffer.push(merge.rank);
            } else {
                level_buffer.push(u8::MAX);
                merge_buffer.push(u32::MAX);
                merge_priority_buffer.push(u16::MAX);
            }
        }

        for level in 0..self.levels {
            let mut index = 0;
            let mut fill_index = 0;
            let mut recalculate_last_merge = false;
            let original_token_buffer_len = token_buffer.len();
            while index < original_token_buffer_len {
                if original_token_buffer_len - index < MASK_SIZE {
                    break;
                }
                let levels = unsafe { level_buffer.get_unchecked_mut(index..) };
                let levels = Simd::from_slice(levels);
                let min_level = levels.reduce_min();
                if min_level >= level {
                    continue;
                }
                let tokens = unsafe { token_buffer.get_unchecked_mut(index..) };
                let tokens = Simd::from_slice(tokens);
                let merges = unsafe { merge_buffer.get_unchecked_mut(index..) };
                let merges = Simd::from_slice(merges);
                let merge_priority = unsafe { merge_priority_buffer.get_unchecked_mut(index..) };
                let merge_priority = Simd::from_slice(merge_priority);

                let result = tokenize::<MASK_SIZE>(tokens, merges, levels, merge_priority, level);

                // Update any merges as needed
                let middle_arr = tokens.as_array();
                let mut update_mask = result.recalculate_mask;
                for i in 0..result.new_tokens_len as usize {
                    // Add the token to the buffer
                    let token = result.new_tokens[i];
                    token_buffer[fill_index] = token;
                    if recalculate_last_merge {
                        let last = unsafe { *middle_arr.get_unchecked(fill_index - 1) };
                        if let Some(merge) = self.passes_might_merge_table.get(last, token) {
                            level_buffer[fill_index - 1] = merge.level;
                            merge_buffer[fill_index - 1] = merge.new_token;
                            merge_priority_buffer[fill_index - 1] = merge.rank;
                        }
                    }
                    let recalculate_merge = update_mask & 1 == 1;
                    if !recalculate_merge {
                        level_buffer[fill_index] = result.new_levels[i];
                        merge_buffer[fill_index] = result.new_merges[i];
                        merge_priority_buffer[fill_index] = result.new_merge_priority[i];
                    }
                    recalculate_last_merge = recalculate_merge;
                    update_mask >>= 1;
                    fill_index += 1;
                }
                index += result.tokens_processed as usize;
            }
        }
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
    fn get(&self, input: u32, output: u32) -> Option<&MergePriority> {
        let (index, key) = Self::index(input, output);

        let value = unsafe { self.mod_bitset.get_unchecked(index as usize) };

        let slice = &self.short_bump[value.ptr as usize..value.end as usize];
        slice.iter().find(|item| item.key == key)
    }
}

#[test]
fn test_empty_merge_table() {
    let table = MergeTable::new(std::iter::empty());
    for x in 0..SIZE {
        for y in 0..SIZE {
            let out = table.get(x, y);
            assert!(out.is_none());
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
                let out = table.get(x, y);
                assert!(out.is_none());
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
                assert_eq!(result, merge_priority);
            } else {
                let out = table.get(x, y);
                assert!(out.is_none());
            }
        }
    }
}
