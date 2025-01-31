#[cfg(test)]
use pretty_assertions::assert_eq;

#[derive(Clone, Copy)]
struct BitMergePattern {
    table: [[u8; 2]; 2],
}

fn generate_table() {
    let table = [BitMergePattern { table: [[0; 2]; 2] }; u8::MAX as usize];

    for pattern in 0..=u8::MAX {}
}

fn construct_merge_pattern(
    pattern: u8,
    following_odd_decreasing_run: bool,
    proceeding_odd_increasing_run: bool,
) -> u8 {
    let mut increasing_run_start = None;
    let mut even_decreasing_run_index = !following_odd_decreasing_run;
    let mut output = 0;
    let mut set_bit = |bit_idx: usize| {
        if bit_idx == 0 || (output >> (bit_idx - 1)) & 1 == 0 {
            output |= 1 << bit_idx;
        }
    };
    for bit_idx in 0..8 {
        let bit = ((pattern >> bit_idx) & 1) != 0;

        match (increasing_run_start, bit) {
            (None, true) => {
                increasing_run_start = Some(bit_idx);
            }
            (Some(_), true) => {}
            (Some(start), false) => {
                let end = bit_idx;
                for bit_idx in (start..end).rev().step_by(2) {
                    set_bit(bit_idx);
                }
                increasing_run_start = None;
                even_decreasing_run_index = false;
                set_bit(bit_idx);
            }
            (None, false) => {
                if even_decreasing_run_index {
                    set_bit(bit_idx);
                }
                even_decreasing_run_index = !even_decreasing_run_index;
            }
        }
    }

    if let Some(start) = increasing_run_start {
        let end = 8;
        for bit_idx in (start..end)
            .rev()
            .skip(proceeding_odd_increasing_run as usize)
            .step_by(2)
        {
            output |= 1 << bit_idx;
        }
    }

    output
}

#[test]
fn test_construct_merge_pattern() {
    let out = construct_merge_pattern(0b0000_0000, false, false);
    println!("{:08b}", out);
    assert_eq!(out, 0b01010101);

    let out = construct_merge_pattern(0b0000_0000, true, false);
    println!("{:08b}", out);
    assert_eq!(out, 0b10101010);
    println!();

    let out = construct_merge_pattern(0b1011_1011, false, false);
    println!("{:08b}", out);
    assert_eq!(out, 0b10101010);
    let out = construct_merge_pattern(0b1011_1011, false, true);
    println!("{:08b}", out);
    assert_eq!(out, 0b00101010);
}

#[test]
fn fuzz_construct_merge_pattern_single() {
    for _ in 0..100 {
        let rand: [u8; 8] = std::array::from_fn(|_| rand::random::<u8>());
        let increasing: Vec<_> = rand
            .iter()
            .zip(rand.iter().skip(1))
            .map(|(a, b)| a > b)
            .collect();
        println!("{increasing:?}");
        let bits = increasing
            .iter()
            .enumerate()
            .fold(0, |acc, (idx, b)| acc | ((*b as u8) << (7 - idx)));

        println!("{bits:08b}");

        let pattern = construct_merge_pattern(bits, false, false);
        println!("{pattern:08b}");

        let mut as_bools = [false; 8];
        for i in 0..8 {
            as_bools[i] = pattern & (1 << (7 - i)) != 0;
        }

        println!("{as_bools:?}");

        let verified_merge_pattern = {
            let mut merges_left = (0..8).collect::<Vec<_>>();
            let mut merges = [false; 8];
            while let Some(&best_merge) = merges_left.iter().max_by_key(|merge| rand[**merge]) {
                merges_left.retain(|&merge| {
                    (merge as isize - best_merge as isize).abs() > 1
                });
                merges[best_merge] = true;
            }
            merges
        };

        println!("{verified_merge_pattern:?}");

        assert_eq!(as_bools, verified_merge_pattern);
    }
}

#[test]
fn fuzz_construct_merge_pattern() {
    for _ in 0..100 {
        let rand: [u8; 32] = std::array::from_fn(|_| rand::random::<u8>());
        let increasing: Vec<_> = rand
            .iter()
            .zip(rand.iter().skip(1))
            .map(|(a, b)| a > b)
            .collect();
        println!("{increasing:?}");
        let first_8 = &increasing[..8];
        let first_8_bits = first_8
            .iter()
            .enumerate()
            .fold(0, |acc, (idx, b)| acc | ((*b as u8) << idx));
        let second_8 = &increasing[8..16];
        let second_8_bits = second_8
            .iter()
            .enumerate()
            .fold(0, |acc, (idx, b)| acc | ((*b as u8) << idx));
        let third_8 = &increasing[16..24];
        let third_8_bits = third_8
            .iter()
            .enumerate()
            .fold(0, |acc, (idx, b)| acc | ((*b as u8) << idx));

        println!("{first_8_bits:08b} {second_8_bits:08b} {third_8_bits:08b}");
        let first_8_trailing_zeros = first_8_bits.trailing_zeros();
        let third_8_leading_ones = third_8_bits.leading_ones();

        let following_odd_decreasing_run = first_8_trailing_zeros % 2 != 1;
        let proceeding_odd_increasing_run = third_8_leading_ones % 2 != 1;

        println!("{following_odd_decreasing_run} {proceeding_odd_increasing_run}");
        let pattern = construct_merge_pattern(
            second_8_bits,
            following_odd_decreasing_run,
            proceeding_odd_increasing_run,
        );

        let mut as_bools = [false; 8];
        for i in 0..8 {
            as_bools[i] = pattern & (1 << i) != 0;
        }

        println!("{as_bools:?}");

        let verified_merge_pattern = {
            let mut merges_left = (0..32).collect::<Vec<_>>();
            let mut merges = [false; 32];
            while let Some(&best_merge) = merges_left.iter().max_by_key(|merge| rand[**merge]) {
                merges_left.retain(|&merge| {
                    (merge as isize - best_merge as isize).abs() > 1
                });
                merges[best_merge] = true;
            }
            merges
        };

        let middle = &verified_merge_pattern[8..16];

        println!("{middle:?}");

        assert_eq!(as_bools, middle);
    }
}

// Variants
// |                                       | following even length decreasing run | following odd length decreasing run |
// |---------------------------------------|--------------------------------------|-------------------------------------|
// | proceeding even length increasing run |                                      |                                     |
// | proceeding odd length increasing run  |                                      |                                     |

// 0011 1011

// Bloom filter for merges?
// Construct perfect bloom filter when we build the table?
