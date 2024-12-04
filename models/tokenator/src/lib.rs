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
