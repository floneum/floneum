#![feature(portable_simd)]
use std::{
    marker::PhantomData,
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        num::SimdUint,
        ptr::SimdConstPtr,
        simd_swizzle, LaneCount, Mask, MaskElement, Simd, SimdElement, SupportedLaneCount, Swizzle,
    },
    u8,
};

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

#[inline]
unsafe fn merge<const N: usize, const M: usize, E: SimdElement + Default>(
    all: [(Simd<E, 8>, u8); N],
) -> (Simd<E, M>, u8)
where
    LaneCount<M>: SupportedLaneCount,
    LaneCount<N>: SupportedLaneCount,
{
    const {
        assert!(N * 8 == M);
    }

    let mut output = Simd::from_array([E::default(); M]);
    let mut offset = 0;
    for (all, this_offset) in all {
        all.copy_to_slice(unsafe { output.as_mut_array().get_unchecked_mut(offset as usize..) });
        offset += this_offset;
    }
    (output, offset)
}

impl KeepValuesImpl<8> for () {
    fn keep_values<S: SimdElement + Default>(
        mask: Mask<S::Mask, 8>,
        values: Simd<S, 8>,
    ) -> (Simd<S, 8>, u8) {
        let mask = mask.to_bitmask() as u8;
        unsafe {
            let (indexes, elements) = KEEP_VALUES_TABLE.get_unchecked((mask & 0b01111111) as usize);
            let simd = Simd::gather_ptr(Simd::splat(values.as_array().as_ptr()).wrapping_add(indexes.cast()));
    
            (simd, elements + (mask >> 7))
        }
    }

    fn keep_values_idx<M: MaskElement>(mask: Mask<M, 8>) -> (Simd<u8, 8>, u8) {
        let mask = mask.to_bitmask();
        let (swizzle, elements) = KEEP_VALUES_TABLE[(mask & 0b01111111) as usize];
        (swizzle, elements + mask as u8 >> 7)
    }
}

macro_rules! copy_8_values {
    ([$($i:expr),+], $values:expr, $mask:expr) => {{
        unsafe {
            merge(
                {
                    [
                        $(
                            {
                                const SIMD_IDX: [usize; 8] = [$i, $i + 1, $i + 2, $i + 3, $i + 4, $i + 5, $i + 6, $i + 7];
                                let chunk = simd_swizzle!($values, SIMD_IDX);
                                let bits = select_nth_byte_for_table::<$i>($mask);
                                let (indexes, elements) = KEEP_VALUES_TABLE.get_unchecked((bits & 0b01111111) as usize);
                                let simd = Simd::gather_ptr(Simd::splat(chunk.as_array().as_ptr()).wrapping_add(indexes.cast()));

                                (simd, elements + (bits >> 7))
                            }
                        ),+
                    ]
                }
            )
        }
    }};
}

macro_rules! copy_8_values_mask {
    ([$($i:expr),+], $mask:expr) => {
        unsafe {
            merge(
                [
                    $(
                        {
                            let bits = select_nth_byte_for_table::<$i>($mask);
                            let (simd, elements) = KEEP_VALUES_TABLE.get_unchecked((bits & 0b01111111) as usize);
                            (*simd, elements + (bits >> 7))
                        }
                    ),+
                ]
            )
        }
    };
}

static KEEP_VALUES_TABLE: [(Simd<u8, 8>, u8); 128] = keep_values_idx_table();

const fn keep_values_idx_table() -> [(Simd<u8, 8>, u8); 128] {
    let mut table = [(Simd::from_array([0; 8]), 0); 128];
    let mut i = 0;
    while i < 128 {
        table[i] = (
            Simd::from_array(keep(i as u8 + 0b10000000)),
            i.count_ones() as u8,
        );
        i += 1;
    }
    table
}

#[inline]
const fn select_nth_byte_for_table<const BYTE: u8>(byte: u64) -> u8 {
    (byte >> BYTE & 0b11111111) as u8
}

impl KeepValuesImpl<16> for () {
    fn keep_values<S: SimdElement + Default>(
        mask: Mask<S::Mask, 16>,
        values: Simd<S, 16>,
    ) -> (Simd<S, 16>, u8) {
        let mask = mask.to_bitmask();
        copy_8_values!([0, 8], values, mask)
    }

    fn keep_values_idx<M: MaskElement>(mask: Mask<M, 16>) -> (Simd<u8, 16>, u8) {
        let bitmask = mask.to_bitmask();
        copy_8_values_mask!([0, 8], bitmask)
    }
}

impl KeepValuesImpl<32> for () {
    fn keep_values<S: SimdElement + Default>(
        mask: Mask<S::Mask, 32>,
        values: Simd<S, 32>,
    ) -> (Simd<S, 32>, u8) {
        let mask = mask.to_bitmask();
        copy_8_values!([0, 8, 16, 24], values, mask)
    }

    fn keep_values_idx<M: MaskElement>(mask: Mask<M, 32>) -> (Simd<u8, 32>, u8) {
        let bitmask = mask.to_bitmask();
        copy_8_values_mask!([0, 8, 16, 24], bitmask)
    }
}

impl KeepValuesImpl<64> for () {
    fn keep_values<S: SimdElement + Default>(
        mask: Mask<S::Mask, 64>,
        values: Simd<S, 64>,
    ) -> (Simd<S, 64>, u8) {
        let mask = mask.to_bitmask();
        copy_8_values!([0, 8, 16, 24, 32, 40, 48, 56], values, mask)
    }

    fn keep_values_idx<M: MaskElement>(mask: Mask<M, 64>) -> (Simd<u8, 64>, u8) {
        let bitmask = mask.to_bitmask();
        copy_8_values_mask!([0, 8, 16, 24, 32, 40, 48, 56], bitmask)
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

const fn keep_usize(i: u8) -> [usize; 8] {
    let mut swizzle = [0; 8];
    let mut b = 0;
    let mut fill_index = 0;
    while b < 8 {
        if (i >> b) & 1 == 1 {
            swizzle[fill_index] = b;
            fill_index += 1;
        };
        b += 1;
    }
    swizzle
}

const fn keep(i: u8) -> [u8; 8] {
    let mut swizzle = [0; 8];
    let keep_usize = keep_usize(i);
    let mut i = 0;
    while i < 8 {
        swizzle[i] = keep_usize[i] as u8;
        i += 1;
    }
    swizzle
}

/// Move all of the values where the mask is true to the front of the vector and replace the remaining values with the or value
#[inline]
fn idx_before_number<const N: usize>(number: u8) -> Mask<CurrentMaskElement, N>
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
    recalculate_mask: u16,
}

impl<const N: usize> AsRef<[u32]> for TokenizationResult<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn as_ref(&self) -> &[u32] {
        &self.new_tokens.as_array()[..self.new_tokens_len as usize]
    }
}

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

    let mut prev_element_in_level = in_this_level.shift_right::<1>();
    prev_element_in_level.set(0, false);
    let start_of_run_mask = in_this_level & (!prev_element_in_level | this_less_then_next);
    let (starts_indexes_raw, starts_idx) = keep_values_idx(start_of_run_mask);
    let starts_indexes_raw = starts_indexes_raw.cast();

    let mut next_element_in_level = in_this_level.shift_left::<1>();
    next_element_in_level.set(SIZE - 1, true);
    let end_of_run_mask = in_this_level & (!next_element_in_level | last_less_than_this_and);
    let (ends_indexes_raw, ends_idx) = keep_values_idx(end_of_run_mask);
    let ends_indexes_raw = ends_indexes_raw.cast();
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
                new_tokens_len: N as u8,
                tokens_processed,
                recalculate_mask: 0,
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

    let merge_with_next_shift_right = prev_is_merge;
    let keep = copy_from | merge_with_next;
    #[cfg(debug_assertions)]
    {
        // Make sure all of the tokens we processed are included in the new tokenization
        let tokens_used = keep | merge_with_next_shift_right;
        let processed = idx_before_number(tokens_processed);
        let tokens_used_or_unprocessed = tokens_used | !processed;
        assert!(tokens_used_or_unprocessed.all());
    }

    let copied_from_original = tokens;
    let copied_from_merge = merge_with_next.cast().select(merges, copied_from_original);
    let (new_merges, _) = keep_values(keep.cast(), merges);
    let (new_merge_priority, _) = keep_values(keep.cast(), merge_priority);
    let (new_levels, _) = keep_values(keep.cast(), levels);
    let (new_tokens, new_tokens_len) = keep_values(keep.cast(), copied_from_merge);
    #[cfg(debug_assertions)]
    {
        // Make sure there are no u32::MAX tokens in the new tokenization
        let processed = &new_tokens.as_array()[..new_tokens_len as usize];
        assert!(processed.iter().all(|&x| x != u32::MAX), "The final processed tokens should not contain uninitialized u32::MAX tokens. Found {processed:?}");
    }

    let recalculate_mask = 0;

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
    // merge_with_next_priority:    1, 2, 3, 4, 2, 2, 3, 0, 1, 2, 3, 4, 2, 2, 3, 4

    let tokens = tokenize(
        tokens,
        tokens_after_merge_with_next,
        levels,
        merge_with_next_priority,
        level,
    );
    assert_eq!(tokens.as_ref(), [5, 2, 1, 5, 7, 5, 2, 1, 5]);
    assert_eq!(tokens.tokens_processed, 13);
}
