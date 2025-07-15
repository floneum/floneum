use std::{collections::HashMap, path::Path};

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Merge {
    pub rank: u32,
    pub pair: [u32; 2],
    pub new_token: u32,
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

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub vocab: HashMap<Vec<u8>, u32>,
    pub inverse_vocab: HashMap<u32, Vec<u8>>,
    pub merges: Vec<Merge>,
    pub bytes: [u32; 256],
    // pub regex: Regex,
}

impl Tokenizer {
    pub fn load_tokenizer(path: impl AsRef<Path>) -> Self {
        let bytes = std::fs::read(path).expect("Failed to read tokenizer file");
        let json = serde_json::from_slice::<Value>(&bytes).unwrap();
        let pretokenizer = json["pre_tokenizer"].clone();
        let sequence = pretokenizer["pretokenizers"][0].clone();
        let pattern = sequence["pattern"]["Regex"].as_str().unwrap();
        // let regex = Regex::new(pattern).unwrap();
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

        let mut bytes = [0; 256];
        for (i, token) in tokens.iter().enumerate() {
            if let [byte] = token.as_slice() {
                bytes[*byte as usize] = i as u32;
            }
        }

        let mut inverse_vocab = HashMap::new();
        for (token, index) in &vocab {
            inverse_vocab.insert(*index, token.clone());
        }

        Self {
            vocab,
            merges,
            bytes,
            inverse_vocab,
            // regex,
        }
    }
}
