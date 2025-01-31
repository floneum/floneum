#![feature(portable_simd)]
use std::{
    marker::PhantomData,
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        num::{SimdInt, SimdUint},
        ptr::SimdConstPtr,
        LaneCount, Mask, MaskElement, Simd, SimdElement, SupportedLaneCount, Swizzle,
    },
    u8,
};

mod new;

mod load;
pub use load::*;

struct ShiftRightMask<const SHIFT: usize, const N: usize, M: MaskElement>(PhantomData<M>)
where
    LaneCount<N>: SupportedLaneCount;

impl<const SHIFT: usize, const N: usize, M: MaskElement> Swizzle<N> for ShiftRightMask<SHIFT, N, M>
where
    LaneCount<N>: SupportedLaneCount,
{
    const INDEX: [usize; N] = const {
        let mut indicies = [0; N];
        let mut i = 0;
        while i < N {
            indicies[(i + SHIFT) % N] = i;
            i += 1;
        }
        indicies
    };
}

struct ShiftLeftMask<const SHIFT: usize, const N: usize, M: MaskElement>(PhantomData<M>)
where
    LaneCount<N>: SupportedLaneCount;

impl<const SHIFT: usize, const N: usize, M: MaskElement> Swizzle<N> for ShiftLeftMask<SHIFT, N, M>
where
    LaneCount<N>: SupportedLaneCount,
{
    const INDEX: [usize; N] = const {
        let mut indicies = [0; N];
        let mut i = 0;
        while i < N {
            indicies[(N + i - SHIFT) % N] = i;
            i += 1;
        }
        indicies
    };
}

trait MaskSwizzleExt<const N: usize, M: MaskElement>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn shift_right<const SHIFT: usize>(self) -> Mask<M, N>;

    fn shift_left<const SHIFT: usize>(self) -> Mask<M, N>;
}

impl<const N: usize, M: MaskElement> MaskSwizzleExt<N, M> for Mask<M, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn shift_right<const SHIFT: usize>(self) -> Mask<M, N> {
        ShiftRightMask::<SHIFT, N, M>::swizzle_mask(self)
    }

    fn shift_left<const SHIFT: usize>(self) -> Mask<M, N> {
        ShiftLeftMask::<SHIFT, N, M>::swizzle_mask(self)
    }
}

#[allow(unused)]
macro_rules! simd_mask_swizzle {
    (
        $vector:expr, $index:expr $(,)?
    ) => {{
        use ::std::simd::Swizzle;
        struct Impl;
        impl Swizzle<{ $index.len() }> for Impl {
            const INDEX: [usize; { $index.len() }] = $index;
        }
        Impl::swizzle_mask($vector)
    }};
    (
        $first:expr, $second:expr, $index:expr $(,)?
    ) => {{
        use $crate::simd::Swizzle;
        struct Impl;
        impl Swizzle<{ $index.len() }> for Impl {
            const INDEX: [usize; { $index.len() }] = $index;
        }
        Impl::swizzle_mask($first, $second)
    }};
}

type CurrentSIMDElement = u16;
type CurrentMaskElement = i16;
const SIZE: usize = std::mem::size_of::<CurrentSIMDElement>() * 8;

pub struct PreparedKeep<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    indexes: Simd<u8, N>,
    elements: u8,
}

impl<const N: usize> PreparedKeep<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub fn new<const M: usize>(bytes: [u8; M]) -> Self
    where
        LaneCount<M>: SupportedLaneCount,
    {
        const { assert!(M * 8 >= N) };
        let mut indexes = [0; N];
        let mut elements = 0;

        let bytes = Simd::from_array(bytes);
        let count_ones_ptr = Simd::splat(COUNT_ONES.as_ptr());
        let byte_counts = unsafe { Simd::gather_ptr(count_ones_ptr.wrapping_add(bytes.cast())) };
        let idx = bytes & Simd::splat(0b01111111);
        let keep_values_ptr = Simd::splat(KEEP_VALUES_COMPRESSED_PTRS.as_ptr());
        let arrays = keep_values_ptr.wrapping_add(idx.cast());

        for (i, (one_bytes, array_start)) in byte_counts
            .as_array()
            .iter()
            .zip(arrays.as_array().iter())
            .enumerate()
            .take(N / 8)
        {
            let array = unsafe { ***array_start };
            let mut simd = Simd::from_array(array);
            simd += Simd::splat(i as u8 * 8);
            simd.copy_to_slice(unsafe { indexes.get_unchecked_mut(elements as usize..) });
            elements += *one_bytes;
        }

        Self {
            indexes: Simd::from_array(indexes),
            elements,
        }
    }

    pub fn swizzle_values<const N2: usize, S: SimdElement + Default>(
        &self,
        values: Simd<S, N2>,
    ) -> Simd<S, N2>
    where
        LaneCount<N2>: SupportedLaneCount,
    {
        const { assert!(N2 <= N) };
        let indexes = self.indexes.as_array();
        let indexes = indexes as *const [u8; N] as *const [u8; N2];
        let indexes = unsafe { Simd::from_array(*indexes) };
        let chunk_ptr = values.as_array().as_ptr();
        let simd = unsafe { Simd::gather_ptr(Simd::splat(chunk_ptr).wrapping_add(indexes.cast())) };

        simd
    }

    pub fn elements(&self) -> u8 {
        self.elements
    }

    pub fn indexes_simd(&self) -> Simd<u8, N> {
        self.indexes
    }

    pub fn indexes(&self) -> &[u8] {
        &self.indexes.as_array()[..N]
    }
}

#[test]
fn test_keep_values_precomputed() {
    let values = Simd::from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    let mask: Mask<i16, 16> = Mask::from_array([
        true, false, true, true, false, true, true, false, false, false, true, false, false, false,
        true, false,
    ]);
    let precomputed = PreparedKeep::<16>::new(mask.to_bitmask().to_le_bytes());
    let output = precomputed.swizzle_values(values);
    let fill = precomputed.elements;
    let as_slice = &output.as_array()[..fill as usize];
    assert_eq!(as_slice, [1, 3, 4, 6, 7, 11, 15]);

    for _ in 0..100 {
        let rand = rand::random::<u64>();
        let bitset: Mask<CurrentMaskElement, 16> = Mask::from_bitmask(rand);
        let precomputed = PreparedKeep::<16>::new(bitset.to_bitmask().to_le_bytes());
        let values: Simd<u16, 16> = Simd::from_array(std::array::from_fn(|_| {
            rand::random::<CurrentSIMDElement>()
        }));
        let output = precomputed.swizzle_values(values);
        let fill = precomputed.elements;
        let as_slice = &output.as_array()[..fill as usize];
        assert_eq!(
            as_slice,
            bitset
                .to_array()
                .iter()
                .enumerate()
                .filter(|(_, b)| **b)
                .map(|(i, _)| values.as_array()[i])
                .collect::<Vec<_>>()
                .as_slice()
        );
    }
}

/// Move all of the values where the mask is true to the front of the vector and replace the remaining values with the or value
#[inline]
pub fn keep_values<const N: usize, S: SimdElement + Default>(
    mask: Mask<S::Mask, N>,
    values: Simd<S, N>,
) -> (Simd<S, N>, u8)
where
    LaneCount<N>: SupportedLaneCount,
    (): KeepValuesImpl<N>,
{
    <()>::keep_values(mask, values)
}

/// Create a simd vector with all of indexes of the true values in the mask
pub fn keep_values_idx<const N: usize, M: MaskElement>(mask: Mask<M, N>) -> (Simd<u8, N>, u8)
where
    LaneCount<N>: SupportedLaneCount,
    (): KeepValuesImpl<N>,
{
    <()>::keep_values_idx(mask)
}

pub trait KeepValuesImpl<const N: usize>: sealed::Sealed {
    fn keep_values<S: SimdElement + Default>(
        mask: Mask<S::Mask, N>,
        values: Simd<S, N>,
    ) -> (Simd<S, N>, u8)
    where
        LaneCount<N>: SupportedLaneCount;

    fn keep_values_idx<M: MaskElement>(mask: Mask<M, N>) -> (Simd<u8, N>, u8)
    where
        LaneCount<N>: SupportedLaneCount;
}

mod sealed {
    pub trait Sealed {}

    impl Sealed for () {}
}

impl KeepValuesImpl<8> for () {
    fn keep_values<S: SimdElement + Default>(
        mask: Mask<S::Mask, 8>,
        values: Simd<S, 8>,
    ) -> (Simd<S, 8>, u8) {
        let byte = mask.to_bitmask() as u8;
        unsafe {
            let elements = COUNT_ONES[byte as usize];
            let idx = (byte & 0b01111111) as usize;
            let array = **KEEP_VALUES_COMPRESSED_PTRS.get_unchecked(idx);
            let indexes = Simd::from_array(array);
            let chunk_ptr = values.as_array().as_ptr();
            let simd = Simd::gather_ptr(Simd::splat(chunk_ptr).wrapping_add(indexes.cast()));

            (simd, elements)
        }
    }

    fn keep_values_idx<M: MaskElement>(mask: Mask<M, 8>) -> (Simd<u8, 8>, u8) {
        let byte = mask.to_bitmask() as u8;
        unsafe {
            let elements = COUNT_ONES[byte as usize];
            let idx = (byte & 0b01111111) as usize;
            let array = **KEEP_VALUES_COMPRESSED_PTRS.get_unchecked(idx);
            let simd = Simd::from_array(array);

            (simd, elements)
        }
    }
}

macro_rules! copy_8_values {
    ($i:expr, $values:expr, $mask:expr) => {{
        let keep = PreparedKeep::<$i>::new($mask.to_le_bytes());
        let values = keep.swizzle_values($values);
        (values, keep.elements())
    }};
}

macro_rules! copy_8_values_mask {
    ($i:expr, $mask:expr) => {{
        let keep = PreparedKeep::<$i>::new($mask.to_le_bytes());
        (keep.indexes_simd(), keep.elements())
    }};
}

#[test]
fn compressed_table() {
    let mut idx_table = Vec::new();
    let mut table = Vec::new();
    'o: for i in (128..=255u8).rev() {
        let key = keep(i as u8, 0);
        let important_elements = i.count_ones() as usize;
        let import_part_of_key = &key[..important_elements];
        for (i, window) in table.windows(important_elements).enumerate() {
            if window == import_part_of_key {
                idx_table.insert(0, i as u16);
                continue 'o;
            }
        }

        idx_table.insert(0, table.len() as u16);
        table.extend_from_slice(&import_part_of_key);
    }

    assert_eq!(table.len(), 320);
    assert_eq!(idx_table.len(), 128);

    assert_eq!(&KEEP_VALUES_COMPRESSED[..table.len()], table.as_slice());
    assert_eq!(KEEP_VALUES_IDX_COMPRESSED, idx_table.as_slice());

    for i in 0..=255u8 {
        let key = keep(i as u8, 0);
        let important_elements = i.count_ones() as usize;
        let import_part_of_key = &key[..important_elements];
        let ignore_last_bit = i & 0b01111111;
        let idx = idx_table[ignore_last_bit as usize];
        assert_eq!(
            import_part_of_key,
            &table[idx as usize..idx as usize + important_elements]
        );
    }
}

static TABLES: ([u8; 326], [u16; 128]) = generate_keep_values_compressed_table();
static KEEP_VALUES_COMPRESSED: &[u8; 326] = &TABLES.0;
static KEEP_VALUES_IDX_COMPRESSED: &[u16; 128] = &TABLES.1;
static KEEP_VALUES_COMPRESSED_PTRS: [&'static [u8; 8]; 128] = {
    let mut ptrs = [&[0u8; 8]; 128];

    let mut j = 0;
    while j < 128 {
        let slice = KEEP_VALUES_COMPRESSED
            .split_at(KEEP_VALUES_IDX_COMPRESSED[j] as usize)
            .1;
        let slice = slice.split_at(8).0;
        ptrs[j] = unsafe { &*(slice.as_ptr() as *const [u8; 8]) };
        j += 1;
    }

    ptrs
};
static COUNT_ONES: [u8; 256] = {
    let mut count_ones = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        count_ones[i] = i.count_ones() as u8;
        i += 1;
    }
    count_ones
};

const fn generate_keep_values_compressed_table() -> ([u8; 326], [u16; 128]) {
    let mut idx_table = [0u16; 128];
    let mut idx_table_fill = 0;
    let mut table = [0u8; 326];
    let mut table_fill = 0usize;
    let mut i = 255u8;
    'o: while i > 127 {
        let key = keep(i as u8, 0);
        let important_elements = i.count_ones() as usize;
        let mut window_index = 0;
        while window_index <= table_fill.saturating_sub(important_elements) {
            let mut equal_index = 0;
            let mut equal = true;
            while equal_index < important_elements {
                if table[window_index + equal_index] != key[equal_index] {
                    equal = false;
                    break;
                }
                equal_index += 1;
            }
            if equal {
                idx_table[127 - idx_table_fill] = window_index as u16;
                idx_table_fill += 1;
                i -= 1;
                continue 'o;
            }
            window_index += 1;
        }

        idx_table[127 - idx_table_fill] = table_fill as u16;
        idx_table_fill += 1;

        let mut index = 0;
        while index < important_elements {
            table[table_fill] = key[index];
            table_fill += 1;
            index += 1;
        }

        i -= 1;
    }
    (table, idx_table)
}

impl KeepValuesImpl<16> for () {
    fn keep_values<S: SimdElement + Default>(
        mask: Mask<S::Mask, 16>,
        values: Simd<S, 16>,
    ) -> (Simd<S, 16>, u8) {
        let mask = mask.to_bitmask() as u16;
        copy_8_values!(16, values, mask)
    }

    fn keep_values_idx<M: MaskElement>(mask: Mask<M, 16>) -> (Simd<u8, 16>, u8) {
        let bitmask = mask.to_bitmask() as u16;
        copy_8_values_mask!(16, bitmask)
    }
}

impl KeepValuesImpl<32> for () {
    fn keep_values<S: SimdElement + Default>(
        mask: Mask<S::Mask, 32>,
        values: Simd<S, 32>,
    ) -> (Simd<S, 32>, u8) {
        let mask = mask.to_bitmask() as u32;
        copy_8_values!(32, values, mask)
    }

    fn keep_values_idx<M: MaskElement>(mask: Mask<M, 32>) -> (Simd<u8, 32>, u8) {
        let bitmask = mask.to_bitmask() as u32;
        copy_8_values_mask!(32, bitmask)
    }
}

impl KeepValuesImpl<64> for () {
    fn keep_values<S: SimdElement + Default>(
        mask: Mask<S::Mask, 64>,
        values: Simd<S, 64>,
    ) -> (Simd<S, 64>, u8) {
        let mask = mask.to_bitmask();
        copy_8_values!(64, values, mask)
    }

    fn keep_values_idx<M: MaskElement>(mask: Mask<M, 64>) -> (Simd<u8, 64>, u8) {
        let bitmask = mask.to_bitmask();
        copy_8_values_mask!(64, bitmask)
    }
}

#[test]
fn test_keep_values() {
    let values = Simd::from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    let mask = Mask::from_array([
        true, false, true, true, false, true, true, false, false, false, true, false, false, false,
        true, false,
    ]);
    let (output, fill) = keep_values(mask, values);
    let as_slice = &output.as_array()[..fill as usize];
    assert_eq!(as_slice, [1, 3, 4, 6, 7, 11, 15]);

    for _ in 0..100 {
        let rand = rand::random::<u64>();
        let bitset = Mask::from_bitmask(rand);
        let values = Simd::from_array(std::array::from_fn(|_| {
            rand::random::<CurrentSIMDElement>()
        }));
        let (output, fill) = keep_values::<64, CurrentSIMDElement>(bitset, values);
        let as_slice = &output.as_array()[..fill as usize];
        assert_eq!(
            as_slice,
            bitset
                .to_array()
                .iter()
                .enumerate()
                .filter(|(_, b)| **b)
                .map(|(i, _)| values.as_array()[i])
                .collect::<Vec<_>>()
                .as_slice()
        );
    }
}

const fn keep_usize(i: u8, offset: u8) -> [usize; 8] {
    let mut swizzle = [0; 8];
    let mut b = 0;
    let mut fill_index = 0;
    while b < 8 {
        if (i >> b) & 1 == 1 {
            swizzle[fill_index] = b + offset as usize;
            fill_index += 1;
        };
        b += 1;
    }
    swizzle
}

const fn keep(i: u8, offset: u8) -> [u8; 8] {
    let mut swizzle = [0; 8];
    let keep_usize = keep_usize(i, offset);
    let mut i = 0;
    while i < 8 {
        swizzle[i] = keep_usize[i] as u8;
        i += 1;
    }
    swizzle
}

/// Move all of the values where the mask is true to the front of the vector and replace the remaining values with the or value
#[inline]
fn idx_before_number<const N: usize, M: MaskElement>(number: u8) -> Mask<M, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    Mask::from_bitmask(u64::MAX >> (64 - number))
}

#[derive(Debug)]
struct TokenizationResult<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    new_tokens: Simd<u32, N>,
    new_merges: Simd<u32, N>,
    new_merge_priority: Simd<u16, N>,
    new_levels: Simd<u8, N>,
    new_tokens_len: u8,
    tokens_processed: u8,
    recalculate_mask: Mask<i8, N>,
}

impl<const N: usize> AsRef<[u32]> for TokenizationResult<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn as_ref(&self) -> &[u32] {
        &self.new_tokens.as_array()[..self.new_tokens_len as usize]
    }
}

// let mut merges = vec![TokenAction::Copy; tokens.len()];

// let mut run_start = 0;
// let mut i = 0;
// while i < increasing_run_starts.len() {
//     let increasing_run_end = increasing_run_starts[i];
//     if increasing_run_end {
//         let mut j = i;
//         while j >= run_start {
//             merges[j] = TokenAction::MergeWithNext;
//             if let Some(next) = merges.get_mut(j + 1) {
//                 *next = TokenAction::Delete;
//             }
//             j = j.saturating_sub(2);
//             if j == 0 {
//                 break;
//             }
//         }
//         run_start = i + 2;
//         i += 2;
//     }
//     else {
//         i += 1;
//     }
// }

const fn keep_increasing_runs<const N: usize>(value: [bool; N], first_already_used: bool, last_value_already_used: bool) -> [bool; N] {
    let mut equivalent_sequence = [0; N];
    let mut i = 0;
    let mut last_value = 128;
    while i < N {
        equivalent_sequence[i] = last_value;
        let greater_than_last = value[i];
        if greater_than_last {
            last_value += 1;
        } else {
            last_value -= 1;
        }
        i += 1;
    }

    let mut merge_with_next = [false; N];
    let mut indexes_used = [false; N]; 
    indexes_used[0] = first_already_used;
    indexes_used[N - 1] = last_value_already_used;
    loop {
        let mut max_index = None;
        {
            let mut i = 0;
            while i < N {
                if indexes_used[i] {
                    i += 1;
                    continue;
                }
                match max_index {
                    None => {
                        max_index = Some(i);
                    }
                    Some(current_max_index) => {
                        if equivalent_sequence[i] > equivalent_sequence[current_max_index] {
                            max_index = Some(i);
                        }
                    }
                }
                i += 1;
            }
        }
        
        let Some(max_index) = max_index else {
            break;
        };
        
        indexes_used[max_index] = true;
        let next_index = max_index + 1;
        if next_index < N {
            indexes_used[next_index] = true;
        }
        if let Some(prev_index) = max_index.checked_sub(1) {
            indexes_used[prev_index] = true;
        }

        merge_with_next[max_index] = true;
    }

    merge_with_next
}

#[test]
fn test_keep_increasing_runs() {
    let mask = keep_increasing_runs([true, true, true, false, false, true, false, true], false, false); 
    assert_eq!(mask, [false, true, false, true, false, false, true, false]);

    let mask = keep_increasing_runs([true; 8], false, false);
    assert_eq!(mask, [false, true, false, true, false, true, false, true]);

    let mask = keep_increasing_runs([false; 8], false, false);
    assert_eq!(mask, [true, false, true, false, true, false, true, false]);

    for _ in 0..100 {
        let random: [bool; 16] = std::array::from_fn(|_| rand::random::<bool>());

        println!("{random:?}");
        let mut streaming_mask = [false; 16];
        let mut i = 15;
        loop {
            let mut run_end = i;
            while run_end > 0 && !random[run_end - 1] {
                run_end -= 1;
            }
            if run_end != i {
                let mut pos = run_end;
                while pos < i {
                    streaming_mask[pos] = true;
                    pos += 2;
                }
                i = run_end;
            } else {
                streaming_mask[i] = true;
                i = i.saturating_sub(2);
            }
            if i == 0 {
                break;
            }
        }

        let mut first_half: [bool; 8] = [false; 8];
        first_half.copy_from_slice(&random[..8]);
        let mut first_numbers = [0u8; 8];
        first_half.iter().enumerate().fold(0, |acc, (idx, b)| {
            first_numbers[idx] = acc;
            acc + (*b as u8)
        });
        let mut second_half: [bool; 8] = [false; 8];
        second_half.copy_from_slice(&random[8..]);
        let mut second_numbers = [0u8; 8];
        second_half.iter().enumerate().fold(0, |acc, (idx, b)| {
            second_numbers[idx] = acc;
            acc + (*b as u8)
        });
        let mut numbers = [0u8; 16];
        random.iter().enumerate().fold(0, |acc, (idx, b)| {
            numbers[idx] = acc;
            acc + (*b as u8)
        });

        let trailing_falses = first_half.iter().rev().take_while(|&&b| !b).count(); 
        let trailing_trues = first_half.iter().rev().take_while(|&&b| b).count();
        let leading_falses = second_half.iter().take_while(|&&b| !b).count();
        let leading_trues = second_half.iter().take_while(|&&b| b).count();

        let mask = keep_increasing_runs(random, false, false);
        let mut first_half_mask = keep_increasing_runs(first_half, false, false);
        let last = *first_half_mask.last().unwrap();
        println!("{:?}", first_half.first().unwrap());
        let second_half_mask = keep_increasing_runs(second_half, false, false);
        if last && (first_half[7] && !second_half[0]) {
            first_half_mask = keep_increasing_runs(first_half, false, true);
            assert!(!first_half_mask.last().unwrap());
        }
        let first_eq = first_half_mask == mask[..8];
        let second_eq = second_half_mask == mask[8..];
        let first = second_half_mask.first().unwrap();
        let gt = last.cmp(&first);
        assert_eq!(streaming_mask, mask);
        if !(first_eq && second_eq) {
            println!("{first_eq} {second_eq} {gt:?} {last:?} {first:?}");
            println!("{first_half:?} {second_half:?}");
            println!("{first_numbers:?} {second_numbers:?}");
            println!("{numbers:?}");
            println!("{first_half_mask:?} {second_half_mask:?}");
            println!("{mask:?}");
            println!("{trailing_falses} {trailing_trues} {leading_falses} {leading_trues}");
            // assert!(first_eq || second_eq);
        }
    }
}

#[test]
fn test_merge_increasing_runs() {
    // // 1) The merge sequence isn't influenced by the exact values of any of the tokens, just the relative values
    // // Merges(S) == Merges(S+K) where S is a sequence and K is a constant
    // // 2) Merges(S1 cat S2) == Merges(S1) cat Merges(S2[last(S1) > first(S2)..])?
    // for _ in 0..100 {
    //     let first_half: [bool; 8] = std::array::from_fn(|_| rand::random::<bool>());
    //     let second_half: [bool; 8] = std::array::from_fn(|_| rand::random::<bool>());
    //     let mut merged = [false; 16];
    //     for i in 0..8 {
    //         merged[i] = first_half[i];
    //         merged[i + 8] = second_half[i];
    //     }
    //     let first_mask = keep_increasing_runs(first_half, false);  
    //     let second_mask = keep_increasing_runs(second_half, *first_mask.last().unwrap());
    //     let mut merged_mask = [false; 16];
    //     for i in 0..8 {
    //         merged_mask[i] = first_mask[i];
    //         merged_mask[i + 8] = second_mask[i];
    //     }
    //     let expected_merged_mask = keep_increasing_runs(merged, false);
    //     assert_eq!(merged_mask, expected_merged_mask);
    // }

    // masking - see runs and move them around like existing system?
}

static INCREASING_MAP_TO_MERGE: [u8; 256] = {
    let mut idx_table = [0; 256];
    let mut i = 0;
    while i <= u8::MAX as usize {
        let mut bits = [false; 8];
        let mut j = 0;
        while j < 8 {
            bits[j] = i & (1 << j) != 0;
            j += 1;
        }
        let bits = keep_increasing_runs(bits, false, false);
        let mut bitset = 0;
        let mut j = 0;
        while j < 8 {
            if bits[j] {
                bitset |= 1 << j;
            }
            j += 1;
        }
        idx_table[i] = bitset;
        i += 1;
    }
    idx_table
};

fn tokenize<const N: usize>(
    tokens: Simd<u32, N>,
    merges: Simd<u32, N>,
    levels: Simd<u8, N>,
    merge_priority: Simd<u16, N>,
    level: u8,
) -> TokenizationResult<N>
where
    LaneCount<N>: SupportedLaneCount,
    (): KeepValuesImpl<N>,
{
    const {
        assert!(N == std::mem::size_of::<CurrentSIMDElement>() * 8);
    }

    let in_this_level: Mask<CurrentMaskElement, N> = levels.simd_eq(Simd::splat(level)).cast();

    let prev_priority = merge_priority.rotate_elements_left::<1>();
    let last_less_than_this = prev_priority.simd_le(merge_priority).cast();
    let mut last_less_than_this_and = last_less_than_this & in_this_level;
    let this_less_then_next = last_less_than_this_and.shift_right::<1>();
    last_less_than_this_and.set(SIZE - 1, false);

    let mut next_element_in_level = in_this_level.shift_left::<1>();
    next_element_in_level.set(SIZE - 1, true);
    let mut end_of_run_mask = in_this_level & (!next_element_in_level | last_less_than_this_and);
    let prev_element_ends_run = end_of_run_mask.shift_right::<1>();
    end_of_run_mask &= !prev_element_ends_run;
    let (ends_indexes_raw, ends_idx) = keep_values_idx(end_of_run_mask);
    let ends_indexes_raw = ends_indexes_raw.cast();
    tracing::trace!("Runs of merges in this level end at:   {:?}", &ends_indexes_raw.as_array()[..ends_idx as usize]);

    let mut prev_element_in_level = in_this_level.shift_right::<1>();
    prev_element_in_level.set(0, false);
    let mut start_of_run_mask = in_this_level & (!prev_element_in_level | this_less_then_next);
    // If we have runs with an end that is right before the start, the previous merge may eat the first token of the next run
    let both = start_of_run_mask & prev_element_ends_run;
    tracing::trace!("before start_of_run_mask: {start_of_run_mask:?}");
    start_of_run_mask |= both.shift_right::<1>();
    start_of_run_mask &= !prev_element_ends_run;
    tracing::trace!("after start_of_run_mask:  {start_of_run_mask:?}");

    let (starts_indexes_raw, starts_idx) = keep_values_idx(start_of_run_mask);
    let starts_indexes_raw = starts_indexes_raw.cast();
    tracing::trace!("Runs of merges in this level start at: {:?}", &starts_indexes_raw.as_array()[..starts_idx as usize]);

    // If there is a final sequence that starts but is not ended, don't include that sequence in this batch. We
    // need to know the length before we merge anything.
    let tokens_processed;
    let starts_idx = if ends_idx == starts_idx {
        tokens_processed = N as u8;
        starts_idx
    } else {
        tokens_processed = starts_indexes_raw[(starts_idx - 1) as usize] as u8;
        tracing::trace!("Chunk ends with increasing priority merges. Only processing up to index {tokens_processed}");
        if ends_idx == 0 {
            return TokenizationResult {
                new_tokens: tokens,
                new_levels: levels,
                new_merges: merges,
                new_merge_priority: merge_priority,
                new_tokens_len: tokens_processed,
                tokens_processed,
                recalculate_mask: Mask::splat(false),
            };
        }
        ends_idx
    };

    if starts_idx == 0 {
        eprintln!("No merges in this chunk!");
        eprintln!("tokens: {tokens:?}");
        eprintln!("levels: {levels:?}");
        eprintln!("merges: {merges:?}");
        eprintln!("merge_priority: {merge_priority:?}");
        eprintln!("starts_idx: {starts_idx}");
        eprintln!("ends_idx: {ends_idx}");
        eprintln!("starts_indexes_raw: {starts_indexes_raw:?}");
        eprintln!("ends_indexes_raw: {ends_indexes_raw:?}");
        unreachable!("This should not be reachable; level: {level}");
    }

    // Create masks that cover all of the indexes in the start and end runs
    let starts_mask = idx_before_number(starts_idx);

    let run_lengths: Simd<_, N> = ends_indexes_raw - starts_indexes_raw;
    let every_other_splat = const {
        let mut num = 0;
        let mut i = 0;
        while i < N {
            if i % 2 == 1 {
                num += 1;
            }
            num <<= 1;
            i += 1;
        }
        Simd::from_array([num; N])
    };
    let run_masks = starts_mask.select(
        every_other_splat >> (Simd::splat(N as CurrentSIMDElement - 1) - run_lengths),
        const { Simd::from_array([CurrentSIMDElement::MIN; N]) },
    );
    let run_masks = run_masks << starts_indexes_raw;

    let even = (run_lengths % Simd::splat(2)).simd_eq(Simd::splat(1)) & starts_mask;
    let copy_from_this_pass = even.select(Simd::splat(1), Simd::splat(0));
    let copy_from_this_pass = copy_from_this_pass << starts_indexes_raw;

    let merge_with_next_bitset = run_masks.reduce_or();
    let merge_with_next = Mask::from_bitmask(merge_with_next_bitset as _);

    let copy_from_this_pass = copy_from_this_pass.reduce_or();
    let copy_from_this_pass = Mask::from_bitmask(copy_from_this_pass as _);

    let all_copy = (!in_this_level) | copy_from_this_pass;
    let mut prev_is_merge = merge_with_next.shift_right::<1>();
    prev_is_merge.set(0, false);
    let processed_in_this_merge = idx_before_number(tokens_processed);
    let copy_from = all_copy & processed_in_this_merge & !prev_is_merge;

    let keep = copy_from | merge_with_next;
    #[cfg(debug_assertions)]
    {
        // Make sure all of the tokens we processed are included in the new tokenization
        let tokens_used = keep | prev_is_merge;
        let processed = idx_before_number(tokens_processed);
        let tokens_used_or_unprocessed = tokens_used | !processed;
        assert!(tokens_used_or_unprocessed.all());
    }

    let copied_from_original = tokens;
    let copied_from_merge = merge_with_next.cast().select(merges, copied_from_original);
    let keeper = PreparedKeep::<N>::new::<2>((keep.to_bitmask() as u16).to_le_bytes());
    let recalculate_mask: Simd<i8, N> = keeper.swizzle_values(merge_with_next.to_int()).cast();
    let recalculate_mask = recalculate_mask.simd_eq(Simd::splat(-1));
    let new_merges = keeper.swizzle_values(merges);
    let new_merge_priority = keeper.swizzle_values(merge_priority);
    let new_levels = keeper.swizzle_values(levels);
    let new_tokens = keeper.swizzle_values(copied_from_merge);
    let new_tokens_len = keeper.elements();

    TokenizationResult {
        new_tokens,
        new_levels,
        new_merges,
        new_merge_priority,
        new_tokens_len,
        tokens_processed,
        recalculate_mask,
    }
}

#[test]
fn test_single_level_merge() {
    _ = tracing_subscriber::fmt::try_init();

    let level = 1;
    let tokens = Simd::from_array([5, 3, 2, 1, 1, 2, 1, 2, 5, 3, 2, 1, 1, 2, 1, 2]);
    let tokens_after_merge_with_next =
        Simd::from_array([1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0]);
    let levels = Simd::from_array([1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2]);
    let merge_with_next_priority =
        Simd::from_array([1, 2, 3, 4, 2, 2, 3, 0, 1, 2, 3, 4, 2, 2, 3, 0]);
    // tokens:                      5, 3, 2, 1, 1, 2, 1, 2
    // token_after_merge_with_next: 1, 2, 3, 4, 5, 6, 7, 0
    // levels:                      1, 1, 2, 2, 1, 1, 1, 2
    // merge_with_next_priority:    1, 2, 3, 4, 2, 2, 3, 0
    // after this level:            5, 2,    1, 5,    7
    //                              0  1  0  0  1  0  1  0
    // 1, 2, x, x, 2, 2, 3, x
    // copy, merge-2, x, merge-2, merge-3

    let tokens = tokenize(
        tokens,
        tokens_after_merge_with_next,
        levels,
        merge_with_next_priority,
        level,
    );
    assert_eq!(tokens.as_ref(), [5, 2, 1, 5, 7, 5, 2, 1, 5, 7]);
}

#[test]
fn test_single_level_merge_with_trailing_increasing_priority() {
    _ = tracing_subscriber::fmt::try_init();

    let level = 1;
    let tokens = Simd::from_array([5, 3, 2, 1, 1, 2, 1, 2, 5, 3, 2, 1, 1, 2, 1, 2]);
    let tokens_after_merge_with_next =
        Simd::from_array([1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0]);
    let levels = Simd::from_array([1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1]);
    let merge_with_next_priority =
        Simd::from_array([1, 2, 3, 4, 2, 2, 3, 0, 1, 2, 3, 4, 2, 2, 3, 4]);
    // tokens:                      5, 3, 2, 1, 1, 2, 1, 2, 5, 3, 2, 1, 1, 2, 1, 2
    // token_after_merge_with_next: 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0
    // levels:                      1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1
    // merge_with_next_priority:    1, 2,  ,  , 2, 2, 3,  , 1, 2,  ,  , 2, 2, 3, 4

    let tokens = tokenize(
        tokens,
        tokens_after_merge_with_next,
        levels,
        merge_with_next_priority,
        level,
    );
    assert_eq!(tokens.as_ref(), [5, 2, 1, 5, 7, 5, 2, 1, 5]);
    assert_eq!(tokens.tokens_processed, 14);
}

#[test]
fn test_trailing_merge() {
    _ = tracing_subscriber::fmt::try_init();

    // level:          0
    // tokens:         [66, 71, 64, 82, 71, 72, 82, 32, 77, 67, 88, 68, 83, 83, 71, 68]
    // levels:         [3, 24, 3, 8, 29, 1, 255, 14, 3, 41, 38, 3, 28, 3, 4, 0]
    // merges:         [331, 4317, 300, 939, 6151, 285, 4294967295, 2127, 303, 10470, 9188, 295, 5683, 339, 383, 261]
    // merge_priority: [45, 614, 8, 80, 582, 26, 65535, 334, 11, 412, 423, 4, 296, 61, 31, 2]
    let level = 0;
    let tokens: Simd<u32, SIZE> = Simd::from_array([
        66, 71, 64, 82, 71, 72, 82, 32, 77, 67, 88, 68, 83, 83, 71, 68,
    ]);
    let merges: Simd<u32, SIZE> = Simd::from_array([
        331, 4317, 300, 939, 6151, 285, 4294967295, 2127, 303, 10470, 9188, 295, 5683, 339, 383,
        261,
    ]);
    let levels: Simd<u8, SIZE> =
        Simd::from_array([3, 24, 3, 8, 29, 1, 255, 14, 3, 41, 38, 3, 28, 3, 4, 0]);
    let merge_priority: Simd<u16, SIZE> = Simd::from_array([
        45, 614, 8, 80, 582, 26, 65535, 334, 11, 412, 423, 4, 296, 61, 31, 2,
    ]);
    let tokens = tokenize(tokens, merges, levels, merge_priority, level);

    let tokens_processed = tokens.tokens_processed as usize;
    let result_len = tokens.new_tokens_len as usize;

    assert_eq!(tokens_processed, SIZE - 1);
    assert_eq!(result_len, SIZE - 1);
    assert_eq!(
        tokens.as_ref(),
        [66, 71, 64, 82, 71, 72, 82, 32, 77, 67, 88, 68, 83, 83, 71]
    );
    assert_eq!(
        tokens.new_levels.as_array()[..result_len],
        [3, 24, 3, 8, 29, 1, 255, 14, 3, 41, 38, 3, 28, 3, 4]
    );
    assert_eq!(
        tokens.new_merges.as_array()[..result_len],
        [331, 4317, 300, 939, 6151, 285, 4294967295, 2127, 303, 10470, 9188, 295, 5683, 339, 383]
    );
    assert_eq!(
        tokens.new_merge_priority.as_array()[..result_len],
        [45, 614, 8, 80, 582, 26, 65535, 334, 11, 412, 423, 4, 296, 61, 31]
    );
    assert_eq!(tokens.tokens_processed as usize, SIZE - 1);
    assert!(!tokens.recalculate_mask.any());
    assert_eq!(tokens.new_tokens_len as usize, SIZE - 1);
}
