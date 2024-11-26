#![feature(portable_simd)]
use std::{
    marker::PhantomData,
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        num::SimdUint,
        simd_swizzle, LaneCount, Mask, MaskElement, Simd, SimdElement, SupportedLaneCount, Swizzle,
    },
    u8,
};

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
fn keep_values<const N: usize, S: SimdElement + Default>(
    mask: Mask<S::Mask, N>,
    values: Simd<S, N>,
) -> (Simd<S, N>, u8)
where
    LaneCount<N>: SupportedLaneCount,
    (): KeepValuesImpl<N>,
{
    <()>::keep_values(mask, values)
}

trait KeepValuesImpl<const N: usize> {
    fn keep_values<S: SimdElement + Default>(
        mask: Mask<S::Mask, N>,
        values: Simd<S, N>,
    ) -> (Simd<S, N>, u8)
    where
        LaneCount<N>: SupportedLaneCount;
}

impl KeepValuesImpl<8> for () {
    fn keep_values<S: SimdElement + Default>(
        mask: Mask<S::Mask, 8>,
        values: Simd<S, 8>,
    ) -> (Simd<S, 8>, u8) {
        keep_values_u8_inner(mask, values)
    }
}

macro_rules! copy_8_values {
    ($i:expr, $fill:expr, $values:expr, $mask:expr, $output:expr) => {{
        let (filled, elements) = {
            const IDX: [usize; 8] = [$i, $i + 1, $i + 2, $i + 3, $i + 4, $i + 5, $i + 6, $i + 7];
            let chunk = simd_swizzle!($values, IDX);
            let mask = simd_mask_swizzle!($mask, IDX);
            keep_values_u8_inner(mask, chunk)
        };
        filled.copy_to_slice(&mut $output.as_mut_array()[$fill as usize..]);
        $fill + elements
    }};
}

impl KeepValuesImpl<16> for () {
    fn keep_values<S: SimdElement + Default>(
        mask: Mask<S::Mask, 16>,
        values: Simd<S, 16>,
    ) -> (Simd<S, 16>, u8) {
        let mut output = Simd::splat(Default::default());
        let fill = 0;
        let fill = copy_8_values!(0, fill, values, mask, output);
        let fill = copy_8_values!(8, fill, values, mask, output);
        (output, fill)
    }
}

impl KeepValuesImpl<32> for () {
    fn keep_values<S: SimdElement + Default>(
        mask: Mask<S::Mask, 32>,
        values: Simd<S, 32>,
    ) -> (Simd<S, 32>, u8) {
        let mut output = Simd::splat(Default::default());
        let fill = 0;
        let fill = copy_8_values!(0, fill, values, mask, output);
        let fill = copy_8_values!(8, fill, values, mask, output);
        let fill = copy_8_values!(16, fill, values, mask, output);
        let fill = copy_8_values!(24, fill, values, mask, output);
        (output, fill)
    }
}

impl KeepValuesImpl<64> for () {
    fn keep_values<S: SimdElement + Default>(
        mask: Mask<S::Mask, 64>,
        values: Simd<S, 64>,
    ) -> (Simd<S, 64>, u8) {
        let mut output = Simd::splat(Default::default());
        let fill = 0;
        let fill = copy_8_values!(0, fill, values, mask, output);
        let fill = copy_8_values!(8, fill, values, mask, output);
        let fill = copy_8_values!(16, fill, values, mask, output);
        let fill = copy_8_values!(24, fill, values, mask, output);
        let fill = copy_8_values!(32, fill, values, mask, output);
        let fill = copy_8_values!(40, fill, values, mask, output);
        let fill = copy_8_values!(48, fill, values, mask, output);
        let fill = copy_8_values!(56, fill, values, mask, output);
        (output, fill)
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

#[inline]
fn keep_values_u8_inner<S: SimdElement>(
    mask: Mask<S::Mask, 8>,
    values: Simd<S, 8>,
) -> (Simd<S, 8>, u8) {
    const fn keep(i: u8) -> [usize; 8] {
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

    macro_rules! gen_table {
        ($($num:expr)*) => {
            fn swizzle_values<S: SimdElement>(values: Simd<S, 8>, bitmask: u8) -> Simd<S, 8> {
                match bitmask {
                    $(
                        $num => {
                            const TABLE: [usize; 8] = keep($num);
                            simd_swizzle!(values, TABLE)
                        },
                    )*
                }
            }
        };
    }
    gen_table!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255);

    let bitmask = mask.to_bitmask();
    let swizzle = swizzle_values(values, bitmask as u8);
    let elements = bitmask.count_ones() as u8;
    (swizzle, elements)
}

/// Move all of the values where the mask is true to the front of the vector and replace the remaining values with the or value
#[inline]
fn idx_before_number<const N: usize>(number: u8) -> Mask<CurrentMaskElement, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    Mask::from_bitmask(u64::MAX >> (64 - number))
}

fn tokenize<const N: usize>(
    tokens: Simd<u32, N>,
    tokens_after_merge_with_next: Simd<u32, N>,
    levels: Simd<u8, N>,
    merge_with_next_priority: Simd<u16, N>,
    level: u8,
) -> (Simd<u32, N>, u8)
where
    LaneCount<N>: SupportedLaneCount,
    (): KeepValuesImpl<N>,
{
    const {
        assert!(N == std::mem::size_of::<CurrentSIMDElement>() * 8);
    }

    let indexes: Simd<CurrentSIMDElement, N> = const {
        let mut index = [0; N];
        let mut i = 0;
        while i < SIZE {
            index[i] = i as CurrentSIMDElement;
            i += 1;
        }
        Simd::from_array(index)
    };
    let in_this_level: Mask<CurrentMaskElement, N> = levels.simd_eq(Simd::splat(level)).cast();

    let prev_priority = merge_with_next_priority.rotate_elements_left::<1>();
    let last_less_than_this = prev_priority.simd_le(merge_with_next_priority).cast();
    let mut last_less_than_this_and = last_less_than_this & in_this_level;
    let this_less_then_next = last_less_than_this_and.shift_right::<1>();
    last_less_than_this_and.set(SIZE - 1, false);
    tracing::trace!("increasing {last_less_than_this:?}");

    let mut prev_element_in_level = in_this_level.shift_right::<1>();
    prev_element_in_level.set(0, false);
    let start_of_run_mask = in_this_level & (!prev_element_in_level | this_less_then_next);
    tracing::trace!("starts:    {start_of_run_mask:?}");
    let (starts_indexes_raw, starts_idx) = keep_values(start_of_run_mask, indexes.cast());
    let starts_mask = idx_before_number(starts_idx);

    let mut next_element_in_level = in_this_level.shift_left::<1>();
    next_element_in_level.set(SIZE - 1, true);
    let end_of_run_mask = in_this_level & (!next_element_in_level | last_less_than_this_and);
    tracing::trace!("ends:      {end_of_run_mask:?}");
    let (ends_indexes_raw, ends_idx) = keep_values(end_of_run_mask, indexes.cast());
    let ends_mask = idx_before_number(ends_idx);

    tracing::trace!("starts: {starts_indexes_raw:?}");
    tracing::trace!("ends: {ends_indexes_raw:?}");
    let run_lengths = ends_indexes_raw - starts_indexes_raw;
    tracing::trace!("runs: {run_lengths:?}");
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
    let run_masks = ends_mask.select(
        every_other_splat >> (Simd::splat(N as CurrentSIMDElement - 1) - run_lengths),
        Simd::splat(CurrentSIMDElement::MIN),
    );
    tracing::trace!(
        "merge unshifted: {:?}",
        run_masks
            .as_array()
            .iter()
            .map(|i| format!("{i:016b}"))
            .collect::<Vec<String>>()
    );
    let run_masks = run_masks << starts_indexes_raw;
    tracing::trace!(
        "merge shifted: {:?}",
        run_masks
            .as_array()
            .iter()
            .map(|i| format!("{i:016b}"))
            .collect::<Vec<String>>()
    );

    let even = (run_lengths % Simd::splat(2)).simd_eq(Simd::splat(1)) & starts_mask;
    let copy_from_this_pass = even.select(Simd::splat(1), Simd::splat(0));
    tracing::trace!(
        "copy unshifted: {:?}",
        copy_from_this_pass
            .as_array()
            .iter()
            .map(|i| format!("{i:016b}"))
            .collect::<Vec<String>>()
    );
    let copy_from_this_pass = copy_from_this_pass << starts_indexes_raw;
    tracing::trace!(
        "copy shifted: {:?}",
        copy_from_this_pass
            .as_array()
            .iter()
            .map(|i| format!("{i:016b}"))
            .collect::<Vec<String>>()
    );

    let merge_with_next_bitset = run_masks.reduce_or();
    tracing::trace!("merge_with_next: {merge_with_next_bitset:016b}");
    let merge_with_next = Mask::from_bitmask(merge_with_next_bitset as _);

    let copy_from_this_pass = copy_from_this_pass.reduce_or();
    tracing::trace!("copy_from_this_pass: {copy_from_this_pass:016b}");
    let copy_from_this_pass = Mask::from_bitmask(copy_from_this_pass as _);
    tracing::trace!("copy_from_this_pass: {copy_from_this_pass:?}");

    let all_copy = (!in_this_level) | copy_from_this_pass;
    tracing::trace!("copy from unfiltered: {:016b}", all_copy.to_bitmask());
    let mut prev_is_merge = merge_with_next.shift_right::<1>();
    prev_is_merge.set(0, false);
    tracing::trace!("prev_is_merge:        {:016b}", prev_is_merge.to_bitmask());
    let copy_from = all_copy & !prev_is_merge;
    tracing::trace!("copy from all:        {:016b}", copy_from.to_bitmask());

    let merge_with_next_shift_right = prev_is_merge;
    tracing::trace!("copy this level: {:016b}", all_copy.to_bitmask());
    tracing::trace!("merge with this: {:016b}", merge_with_next.to_bitmask());
    tracing::trace!(
        "merge with next: {:016b}",
        merge_with_next_shift_right.to_bitmask()
    );
    let keep = copy_from | merge_with_next;
    #[cfg(debug_assertions)]
    {
        let new_array_fill = keep | merge_with_next_shift_right;
        tracing::trace!("remaining {:?}", new_array_fill);
        assert!(new_array_fill.all());
    }

    tracing::trace!("copied from: {:?}", copy_from);
    let copied_from_original = copy_from.cast().select(tokens, Simd::splat(u32::MAX));
    tracing::trace!("copied from original: {:?}", copied_from_original);
    let copied_from_merge = merge_with_next
        .cast()
        .select(tokens_after_merge_with_next, copied_from_original);
    tracing::trace!("copied from merge: {:?}", copied_from_merge);
    let (sequential, len) = keep_values(keep.cast(), copied_from_merge);
    let tokens = &sequential.as_array()[..len as usize];
    tracing::trace!("sequential: {:?}", tokens);

    (sequential, len)
}

#[test]
fn testing() {
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

    let (tokens, len) = tokenize(
        tokens,
        tokens_after_merge_with_next,
        levels,
        merge_with_next_priority,
        level,
    );
    let tokens = &tokens.as_array()[..len as usize];
    assert_eq!(tokens, [5, 2, 1, 5, 7, 5, 2, 1, 5, 7]);
}
