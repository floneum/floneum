
use std::ops::Range;
use std::u32;
use std::{
    collections::hash_map::Entry,
    fmt::{Debug, Formatter},
    hash::Hash,
    io::BufRead,
};

use bumpalo::{
    collections::{vec, Vec as BumpVec},
    Bump,
};
use hashbrown::{hash_map::DefaultHashBuilder, HashMap, HashSet};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use trie_rs::map::{Trie, TrieBuilder};

struct FastBPE {
    vocab: Box<[String]>,
}

fn main(){
    let mut trie;
    let mut might_merge_bitset = BitHashMap::new();

    // Map from token to all the tokens that token could turn into
    let mut possibility_map: HashMap<u32, _, _, &Bump> = HashMap::new_in(&bump);
    // Map from token to any merges it is involved in
    let mut merge_with_next: HashMap<u32, Vec<usize, &Bump>> = HashMap::new();
    let mut merge_with_prev: HashMap<u32, Vec<usize, &Bump>> = HashMap::new();
    for (i, merge) in merges.iter().enumerate() {
        merge_with_next
            .entry(merge.pair[0])
            .or_insert_with(|| Vec::new_in(&bump))
            .push(i);
        merge_with_prev
            .entry(merge.pair[1])
            .or_insert_with(|| Vec::new_in(&bump))
            .push(i);
    }

    fn get_cached<'a>(
        bump: &'a Bump,
        possibility_map: &mut HashMap<
            u32,
            HashSet<u32, DefaultHashBuilder, &'a Bump>,
            DefaultHashBuilder,
            &Bump,
        >,
        merge_map: &HashMap<u32, Vec<usize, &'a Bump>>,
        token: u32,
        merges: &[Merge],
        append_to: &mut HashSet<u32, DefaultHashBuilder, &'a Bump>,
    ) {
        // If it is already cached, return the cached value
        if let Some(occupied) = possibility_map.get(&token) {
            append_to.extend(occupied.iter().copied());
            return;
        }

        // Otherwise the max merge length is the max of all the tokens the new token could possibly be merged with
        let mut possible = HashSet::new_in(bump);
        possible.insert(token);
        if let Some(merges_for_new) = merge_map.get(&token) {
            for merge_index in merges_for_new {
                let merge = &merges[*merge_index];
                let new = merge.new_token;

                get_cached(bump, possibility_map, merge_map, new, merges, &mut possible);
            }
        }

        append_to.extend(possible.iter().copied());

        possibility_map.insert(token, possible);
    }

    let mut lookahead_for_token = vec![0; tokens.len()];

    for (token, lookahead_result) in lookahead_for_token.iter_mut().enumerate() {
        let mut result = HashSet::new_in(&bump);
        get_cached(
            &bump,
            &mut possibility_map,
            &merge_with_next,
            token as u32,
            &merges,
            &mut result,
        );
        *lookahead_result = result.len();
    }

    {
        let (max_token, max) = lookahead_for_token
            .iter()
            .enumerate()
            .filter(|(token, _)| tokens[*token].len() == 1)
            .max_by_key(|(_, x)| *x)
            .unwrap();
        let sum = lookahead_for_token
            .iter()
            .enumerate()
            .filter(|(token, _)| tokens[*token].len() == 1)
            .map(|(_, x)| *x)
            .sum::<usize>();

        println!("lookahead info for byte tokens");
        println!("max: {:?}", max);
        println!(
            "max token: {:?}",
            std::str::from_utf8(&tokens[max_token]).unwrap()
        );
        println!("avg {:?}", sum as f32 / 255.);
    }
    {
        let (max_token, max) = lookahead_for_token
            .iter()
            .enumerate()
            .filter(|(token, _)| tokens[*token].len() > 1)
            .max_by_key(|(_, x)| *x)
            .unwrap();
        let sum = lookahead_for_token
            .iter()
            .enumerate()
            .filter(|(token, _)| tokens[*token].len() > 1)
            .map(|(_, x)| *x)
            .sum::<usize>();
        let non_byte_tokens = lookahead_for_token
            .iter()
            .enumerate()
            .filter(|(token, _)| tokens[*token].len() > 1)
            .count();

        println!("lookahead info for non byte tokens");
        println!("max: {:?}", max);
        println!(
            "max token: {:?}",
            std::str::from_utf8(&tokens[max_token]).unwrap()
        );
        println!("avg {:?}", sum as f32 / non_byte_tokens as f32);
    }

    let mut max = None;
    let mut sum = 0;
    let mut byte_tokens = [0; 256];
    for byte in 0..255 {
        let token = tokens.iter().position(|v| *v == [byte]);
        if let Some(token) = token {
            byte_tokens[byte as usize] = token as u32;
            let result = possibility_map.get(&(token as u32)).unwrap();
            let could_merge_with: HashSet<_> = result
                .iter()
                .flat_map(|x| merge_with_next.get(x).into_iter().flatten())
                .map(|x| merges[*x].pair[1])
                .collect();
            sum += could_merge_with.len();
            if let Some((max_len, _)) = max {
                if could_merge_with.len() > max_len {
                    max = Some((could_merge_with.len(), token));
                }
            } else {
                max = Some((could_merge_with.len(), token));
            }
        }
    }
    let (max, max_token) = max.unwrap();

    println!();
    println!("byte tokens could merge with");
    println!("max: {:?}", max);
    println!(
        "max token: {:?}",
        std::str::from_utf8(&tokens[max_token]).unwrap()
    );
    println!("avg {:?}", sum as f32 / 255.);

    let mut could_merge_with: Vec<HashSet<u32>> = vec![HashSet::new(); tokens.len()];

    for (token, possible) in possibility_map.iter() {
        let index = *token as usize;
        if tokens[index].len() <= 1 {
            continue;
        }
        could_merge_with[index] = possible
            .iter()
            .flat_map(|x| merge_with_next.get(x).into_iter().flatten())
            .map(|x| merges[*x].pair[1])
            .collect();
    }

    let max = could_merge_with
        .iter()
        .map(|x| x.len())
        .max()
        .unwrap_or_default();
    let sum = could_merge_with.iter().map(|x| x.len()).sum::<usize>();

    println!();
    println!("non byte tokens could merge with");
    println!("max: {:?}", max);
    println!("avg {:?}", sum as f32 / tokens.len() as f32);

    let size_of_bytes = std::mem::size_of_val(&*might_merge_bitset.mod_bitset);
    let size_of_bits = size_of_bytes * 8;
    println!("size (kb): {:?}", size_of_bytes / 1024);
    println!("size (bits): {:?}", size_of_bits);

    for (first_token, could_merge_with) in could_merge_with.iter().enumerate() {
        for next_token in could_merge_with {
            might_merge_bitset.set(first_token, *next_token as usize, true);
        }
    }

    let one_bits = might_merge_bitset
        .mod_bitset
        .iter()
        .map(|x| x.count_ones())
        .sum::<u32>();
    let real_bit_size = SIZE.pow(2);
    println!();
    println!("occupied bits: {real_bit_size}");
    println!(
        "used bits percentage: {:?}",
        real_bit_size as f32 / size_of_bits as f32 * 100.
    );
    println!("one bits: {one_bits}");
    println!("avg {:?}", one_bits as f32 / real_bit_size as f32);

    // // print the bitset
    // for row in 0..SIZE {
    //     for col in 0..SIZE {
    //         let bit = might_merge_bitset.get(row, col);
    //         if bit {
    //             print!("1");
    //         } else {
    //             print!("0");
    //         }
    //     }
    //     println!();
    // }
    trie = {
        let mut builder = TrieBuilder::new();
        for (i, token) in tokens.iter().enumerate() {
            builder.insert(token.iter().copied(), i as u32);
        }
        builder.build()
    };
}

let std_input = std::io::stdin();
let mut input = std_input.lock();

loop {
    let mut line = String::new();
    input.read_line(&mut line).unwrap();

    #[derive(Clone)]
    struct UnmergedToken {
        value: u32,
        string: String,
    }

    impl Debug for UnmergedToken {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            let value = self.value();
            let resolved = self.resolved();
            f.debug_struct("UnmergedToken")
                .field("value", &value)
                .field("resolved", &resolved)
                .finish()
        }
    }

    impl UnmergedToken {
        fn new(value: u32, string: String) -> Self {
            UnmergedToken { value, string }
        }

        fn resolved(&self) -> bool {
            self.value & const { 1 << 31 } != 0
        }

        fn set_resolved(&mut self, resolved: bool) {
            if resolved {
                self.value |= 1 << 31;
            } else {
                self.value &= !(1 << 31);
            }
        }

        fn value(&self) -> u32 {
            self.value & !(1 << 31)
        }
    }

    let mut unmerged_tokens = Vec::new();
    for token_str in line.split_whitespace() {
        let token = match trie.exact_match(token_str) {
            Some(token) => token,
            None => {
                println!("not found: {:?}", token_str);
                continue;
            }
        };
        let unmerged = UnmergedToken::new(*token, token_str.to_string());
        unmerged_tokens.push(unmerged);
    }

    for [first, second] in unmerged_tokens.array_windows() {
        let first_tok = first.value();
        let second_tok = second.value();

        let value = might_merge_bitset.get(first_tok as usize, second_tok as usize);
        let might_merge = if value {
            "might merge"
        } else {
            "will never merge"
        };
        println!("{:?} {:?} {}", first.string, second.string, might_merge);
    }

    println!("{:?}", unmerged_tokens);
}

type Int = u32;

const INT_SIZE: usize = std::mem::size_of::<Int>() * 8;
const SIZE: usize = INT_SIZE.pow(2) * 2;
const ROW_SIZE: usize = SIZE;
const COL_SIZE: usize = SIZE / INT_SIZE;

struct BitHashMap {
    mod_bitset: Box<[Int; ROW_SIZE * COL_SIZE]>,
}

impl BitHashMap {
    fn new() -> Self {
        Self {
            mod_bitset: Box::new([0; ROW_SIZE * COL_SIZE]),
        }
    }

    fn index(&self, input: usize, output: usize) -> (usize, usize) {
        let input_mod = input % SIZE;
        let output_mod = output % SIZE;
        let inner_index = input_mod / INT_SIZE;
        let index = output_mod * COL_SIZE + inner_index;
        let bit = input_mod % INT_SIZE;
        (index, bit)
    }

    fn set(&mut self, input: usize, output: usize, value: bool) {
        let (index, bit) = self.index(input, output);
        if value {
            self.mod_bitset[index] |= 1 << bit;
        } else {
            self.mod_bitset[index] &= !(1 << bit);
        }
    }

    fn get(&self, input: usize, output: usize) -> bool {
        let (index, bit) = self.index(input, output);
        (self.mod_bitset[index] & (1 << bit)) != 0
    }
}
