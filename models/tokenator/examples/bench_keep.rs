#![feature(portable_simd)]
use std::{hint::black_box, simd::Mask};

use tokenator::*;

fn main() {
    let masks = (0..100)
        .map(|_| Mask::<i16, 16>::from_bitmask(rand::random::<u64>()))
        .collect::<Vec<_>>();
    loop {
        for mask in masks.iter().copied() {
            black_box(keep_values_idx(mask));
        }
    }
}
