#![feature(portable_simd)]
use std::{
    hint::black_box,
    simd::{Mask, Simd},
};

use tokenator::*;

fn main() {
    let masks = (0..100)
        .map(|_| {
            (
                Mask::<i16, 16>::from_bitmask(rand::random::<u64>()),
                Simd::from_array(std::array::from_fn(|_| rand::random::<u16>())),
            )
        })
        .collect::<Vec<_>>();

    loop {
        for (mask, values) in masks.iter().copied() {
            black_box(keep_values(mask, values));
        }
    }
}
