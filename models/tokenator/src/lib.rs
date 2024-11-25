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
        while i < N as usize {
            indicies[(i + SHIFT) % N as usize] = i;
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
        while i < N as usize {
            indicies[(N + i - SHIFT) % N as usize] = i;
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

type CurrentSIMDElement = u8;
type CurrentMaskElement = i8;
const SIZE: usize = 8;
const SIZE_SPACE: usize = 2usize.pow(SIZE as u32);

/// Move all of the values where the mask is true to the front of the vector and replace the remaining values with the or value
#[inline]
fn keep_values<const N: usize>(
    mask: Mask<CurrentMaskElement, N>,
    values: Simd<CurrentSIMDElement, N>,
) -> (Simd<CurrentSIMDElement, N>, Mask<CurrentMaskElement, N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    match N {
        _ => todo!(),
    }
}

#[inline]
fn keep_values_u8_inner(
    mask: Mask<CurrentMaskElement, 8>,
    values: Simd<CurrentSIMDElement, 8>,
) -> (Simd<CurrentSIMDElement, 8>, u8) {
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
            fn swizzle_values(values: Simd<CurrentSIMDElement, 8>, bitmask: u8) -> Simd<CurrentSIMDElement, 8> {
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
fn keep_values_u8(
    mask: Mask<CurrentMaskElement, 8>,
    values: Simd<CurrentSIMDElement, 8>,
) -> (Simd<CurrentSIMDElement, 8>, Mask<CurrentMaskElement, 8>) {
    let (swizzle, elements) = keep_values_u8_inner(mask, values);
    let mask = Mask::from_bitmask((0b11111111 >> (8 - elements)) as u64);  
    (swizzle, mask)
}


const INDEXES: Simd<u8, SIZE> = {
    let mut index = [0; SIZE];
    let mut i = 0;
    while i < SIZE {
        index[i] = i as u8;
        i += 1;
    }
    Simd::from_array(index)
};

#[test]
fn testing() {
    let level = 1;
    let tokens: Simd<CurrentSIMDElement, SIZE> = Simd::from_array([5, 3, 2, 1, 1, 2, 1, 2]);
    let token_after_merge_with_next: Simd<CurrentSIMDElement, SIZE> =
        Simd::from_array([1, 2, 3, 4, 5, 6, 7, 0]);
    let levels: Simd<CurrentSIMDElement, SIZE> = Simd::from_array([1, 1, 2, 2, 1, 1, 1, 2]);
    let merge_with_next_priority: Simd<CurrentSIMDElement, SIZE> =
        Simd::from_array([1, 2, 3, 4, 2, 2, 3, 0]);
    // tokens:                      5, 3, 2, 1, 1, 2, 1, 2
    // token_after_merge_with_next: 1, 2, 3, 4, 5, 6, 7, 0
    // levels:                      1, 1, 2, 2, 1, 1, 1, 2
    // merge_with_next_priority:    1, 2, 3, 4, 2, 2, 3, 0
    // after this level:            5, 2,    1, 5,    7
    //                              0  1  0  0  1  0  1  0
    // 1, 2, x, x, 2, 2, 3, x
    // copy, merge-2, x, merge-2, merge-3

    let in_this_level = levels.simd_eq(Simd::splat(level));

    let prev_priority = merge_with_next_priority.rotate_elements_left::<1>();
    let last_less_than_this: Mask<_, SIZE> = prev_priority.simd_le(merge_with_next_priority);
    let last_less_than_this_and = last_less_than_this & in_this_level;
    let this_less_then_next: Mask<_, SIZE> = last_less_than_this_and.shift_right::<1>();
    println!("increasing {last_less_than_this:?}");

    let mut prev_element_in_level: Mask<CurrentMaskElement, SIZE> =
        in_this_level.shift_right::<1>();
    prev_element_in_level.set(0, false);
    let start_of_run_mask = in_this_level & (!prev_element_in_level | this_less_then_next);
    println!("starts:    {start_of_run_mask:?}");
    let (starts_indexes_raw, starts_mask) = keep_values_u8(start_of_run_mask, INDEXES.cast());
    assert_eq!(
        starts_mask.select(starts_indexes_raw, Simd::splat(CurrentSIMDElement::MAX)),
        Simd::from_array([
            0,
            4,
            5,
            CurrentSIMDElement::MAX,
            CurrentSIMDElement::MAX,
            CurrentSIMDElement::MAX,
            CurrentSIMDElement::MAX,
            CurrentSIMDElement::MAX
        ])
    );

    let mut next_element_in_level = in_this_level.shift_left::<1>();
    next_element_in_level.set(SIZE - 1, false);
    let end_of_run_mask = in_this_level & (!next_element_in_level | last_less_than_this_and);
    println!("ends:      {end_of_run_mask:?}");
    let (ends_indexes_raw, ends_mask) = keep_values_u8(end_of_run_mask, INDEXES.cast());
    assert_eq!(
        ends_mask.select(ends_indexes_raw, Simd::splat(CurrentSIMDElement::MAX)),
        Simd::from_array([
            1,
            4,
            6,
            CurrentSIMDElement::MAX,
            CurrentSIMDElement::MAX,
            CurrentSIMDElement::MAX,
            CurrentSIMDElement::MAX,
            CurrentSIMDElement::MAX
        ])
    );

    println!("starts: {starts_indexes_raw:?}");
    println!("ends: {ends_indexes_raw:?}");
    let run_lengths = ends_indexes_raw - starts_indexes_raw;
    println!("runs: {run_lengths:?}");
    let every_other = const {
        let mut num = 0;
        let mut i = 0;
        while i < SIZE {
            if i % 2 == 1 {
                num += 1;
            }
            num <<= 1;
            i += 1;
        }
        num
    };
    let every_other_splat = Simd::splat(every_other);
    let run_masks = ends_mask.select(
        every_other_splat >> (Simd::splat(SIZE as CurrentSIMDElement - 1) - run_lengths),
        Simd::splat(CurrentSIMDElement::MIN),
    );
    println!(
        "merge unshifted: {:?}",
        run_masks
            .as_array()
            .iter()
            .map(|i| format!("{i:08b}"))
            .collect::<Vec<String>>()
    );
    let run_masks = run_masks << starts_indexes_raw;
    println!(
        "merge shifted: {:?}",
        run_masks
            .as_array()
            .iter()
            .map(|i| format!("{i:08b}"))
            .collect::<Vec<String>>()
    );

    let even = (run_lengths % Simd::splat(2)).simd_eq(Simd::splat(1)) & starts_mask;
    let copy_from_this_pass = even.select(Simd::splat(1), Simd::splat(0));
    println!(
        "copy unshifted: {:?}",
        copy_from_this_pass
            .as_array()
            .iter()
            .map(|i| format!("{i:08b}"))
            .collect::<Vec<String>>()
    );
    let copy_from_this_pass = copy_from_this_pass << starts_indexes_raw;
    println!(
        "copy shifted: {:?}",
        copy_from_this_pass
            .as_array()
            .iter()
            .map(|i| format!("{i:08b}"))
            .collect::<Vec<String>>()
    );

    let merge_with_next_bitset = run_masks.reduce_or();
    // 12xx223x
    // 01001010
    println!("merge_with_next: {merge_with_next_bitset:08b}");
    let merge_with_next = Mask::from_bitmask(merge_with_next_bitset as _);
    assert_eq!(merge_with_next, Mask::from_array([false, true, false, false, true, false, true, false]));

    let copy_from_this_pass = copy_from_this_pass.reduce_or();
    println!("copy_from_this_pass: {copy_from_this_pass:08b}");
    let copy_from_this_pass = Mask::from_bitmask(copy_from_this_pass as _);
    println!("copy_from_this_pass: {copy_from_this_pass:?}");
    assert_eq!(copy_from_this_pass, Mask::from_bitmask(0b00100001));

    let all_copy = (!in_this_level) | copy_from_this_pass;
    println!("copy from unfiltered: {:08b}", all_copy.to_bitmask());
    let mut prev_is_merge = merge_with_next.shift_right::<1>();
    prev_is_merge.set(0, false);
    println!("prev_is_merge:        {:08b}", prev_is_merge.to_bitmask());
    let copy_from = all_copy & !prev_is_merge;
    println!("copy from all:        {:08b}", copy_from.to_bitmask());

    let merge_with_next_shift_right = prev_is_merge;
    println!("copy this level: {:08b}", all_copy.to_bitmask());
    println!("merge with this: {:08b}", merge_with_next.to_bitmask());
    println!(
        "merge with next: {:08b}",
        merge_with_next_shift_right.to_bitmask()
    );
    let keep = copy_from | merge_with_next;
    let new_array_fill = keep | merge_with_next_shift_right;
    println!("remaining {:?}", new_array_fill);
    assert!(new_array_fill.all());

    println!("copied from: {:?}", copy_from);
    let copied_from_original = copy_from.select(tokens, Simd::splat(CurrentSIMDElement::MAX));
    println!("copied from original: {:?}", copied_from_original);
    let copied_from_merge = merge_with_next.select(token_after_merge_with_next, copied_from_original);
    println!("copied from merge: {:?}", copied_from_merge);
    let (sequential, len) = keep_values_u8_inner(keep, copied_from_merge); 
    let tokens = &sequential.as_array()[..len as usize];
    println!("sequential: {:?}", tokens);
    assert_eq!(tokens, [5, 2, 1, 5, 7]);
}
