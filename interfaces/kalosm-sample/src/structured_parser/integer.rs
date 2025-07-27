use crate::bail;

use crate::{
    CreateParserState, EmptyNumber, InvalidSignLocation, LeadingZeroError, OutOfRangeError,
    ParseStatus, Parser,
};
use std::ops::RangeInclusive;

/// A parser for an integer.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct IntegerParser {
    range: RangeInclusive<i128>,
}

impl IntegerParser {
    /// Create a new integer parser.
    pub fn new(range: RangeInclusive<i128>) -> Self {
        if range.start() > range.end() {
            Self {
                range: *range.end()..=*range.start(),
            }
        } else {
            Self { range }
        }
    }
}

impl CreateParserState for IntegerParser {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        IntegerParserState::default()
    }
}

impl IntegerParser {
    fn can_be_negative(&self) -> bool {
        *self.range.start() < 0
    }

    fn is_number_valid(&self, value: i128) -> bool {
        self.range.contains(&value)
    }

    fn should_stop(&self, value: i128) -> bool {
        match value.checked_mul(10) {
            Some(after_next_digit) => {
                (after_next_digit > 0 && after_next_digit > *self.range.end())
                    || (after_next_digit <= 0 && after_next_digit < *self.range.start())
            }
            None => true,
        }
    }

    fn could_number_become_valid(&self, value: i128) -> bool {
        if self.is_number_valid(value) {
            true
        } else {
            let start_value = *self.range.start();
            let end_value = *self.range.end();
            let positive = value >= 0;
            // Check if adding a digit would make the number invalid
            if positive {
                if value * 10 > end_value {
                    return false;
                }
            } else if value * 10 < start_value {
                return false;
            }

            // Check if the digits are within the range so far
            let digits = value.abs().checked_ilog10().map(|x| x + 1).unwrap_or(1);
            let start_digits = start_value
                .abs()
                .checked_ilog10()
                .map(|x| x + 1)
                .unwrap_or(1);
            let end_digits = end_value.abs().checked_ilog10().map(|x| x + 1).unwrap_or(1);
            let mut check_end = true;
            let mut check_start = true;
            for digit in 1..(digits + 1) {
                let selected_digit = value / (10_i128.pow(digits - digit)) % 10;
                let selected_start_digit = start_value / (10_i128.pow(start_digits - digit)) % 10;
                let selected_end_digit = end_value / (10_i128.pow(end_digits - digit)) % 10;

                if check_start {
                    match selected_digit.cmp(&selected_start_digit) {
                        std::cmp::Ordering::Greater => {
                            check_start = false;
                        }
                        std::cmp::Ordering::Less => {
                            return false;
                        }
                        std::cmp::Ordering::Equal => {}
                    }
                }
                if check_end {
                    match selected_digit.cmp(&selected_end_digit) {
                        std::cmp::Ordering::Greater => {
                            return false;
                        }
                        std::cmp::Ordering::Less => {
                            check_end = false;
                        }
                        std::cmp::Ordering::Equal => {}
                    }
                }
            }
            true
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, Default)]
enum IntegerParserProgress {
    #[default]
    Initial,
    AfterSign,
    AfterDigit,
}

impl IntegerParserProgress {
    fn is_after_digit(&self) -> bool {
        matches!(self, IntegerParserProgress::AfterDigit)
    }
}

/// The state of an integer parser.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct IntegerParserState {
    state: IntegerParserProgress,
    value: u64,
    positive: bool,
}

impl Default for IntegerParserState {
    fn default() -> Self {
        IntegerParserState {
            state: IntegerParserProgress::Initial,
            value: 0,
            positive: true,
        }
    }
}

impl Parser for IntegerParser {
    type Output = i128;
    type PartialState = IntegerParserState;

    fn parse<'a>(
        &self,
        state: &IntegerParserState,
        input: &'a [u8],
    ) -> crate::ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        let mut value = state.value;
        let mut positive = state.positive;
        let mut state = state.state;

        for index in 0..input.len() {
            let input_byte = input[index];
            let digit = match input_byte {
                b'0'..=b'9' => {
                    if state == IntegerParserProgress::AfterDigit
                        && value == 0
                        && input_byte == b'0'
                    {
                        bail!(LeadingZeroError);
                    }
                    input_byte - b'0'
                }
                b'-' => {
                    if state == IntegerParserProgress::Initial {
                        state = IntegerParserProgress::AfterSign;
                        positive = false;
                        if !self.can_be_negative() {
                            bail!(OutOfRangeError)
                        }
                        continue;
                    } else {
                        bail!(InvalidSignLocation)
                    }
                }
                _ => {
                    if state.is_after_digit() {
                        let result = value as i128 * if positive { 1 } else { -1 };
                        if self.is_number_valid(result) {
                            return Ok(ParseStatus::Finished {
                                result,
                                remaining: &input[index..],
                            });
                        }
                        bail!(OutOfRangeError)
                    } else {
                        bail!(EmptyNumber)
                    }
                }
            };

            state = IntegerParserProgress::AfterDigit;
            match value.checked_mul(10) {
                Some(v) => value = v + u64::from(digit),
                None => {
                    let signed_value = value as i128 * if positive { 1 } else { -1 };
                    if self.is_number_valid(signed_value) {
                        return Ok(ParseStatus::Finished {
                            result: signed_value,
                            remaining: &input[index..],
                        });
                    }
                    bail!(OutOfRangeError)
                }
            }

            let signed_value = value as i128 * if positive { 1 } else { -1 };

            if self.should_stop(signed_value) {
                return Ok(ParseStatus::Finished {
                    result: signed_value,
                    remaining: &input[index + 1..],
                });
            }

            if !self.could_number_become_valid(signed_value) {
                if self.is_number_valid(signed_value) {
                    return Ok(ParseStatus::Finished {
                        result: signed_value,
                        remaining: &input[index + 1..],
                    });
                }
                bail!(OutOfRangeError)
            }
        }

        Ok(ParseStatus::Incomplete {
            new_state: IntegerParserState {
                state,
                value,
                positive,
            },
            required_next: Default::default(),
        })
    }
}

#[test]
fn integer_parser() {
    for _ in 0..100 {
        let random_number = rand::random::<i64>() as i128;
        let range = random_number.saturating_sub(rand::random::<u8>() as i128)
            ..=random_number.saturating_add(rand::random::<u8>() as i128);
        assert!(range.contains(&random_number));
        println!("range: {range:?}");
        println!("random_number: {random_number:?}");

        let parser = IntegerParser { range };
        let mut state = IntegerParserState::default();

        let mut as_string = random_number.to_string();
        let cap_string = rand::random::<char>().to_string();
        as_string += &cap_string;
        let mut bytes = as_string.as_bytes().to_vec();
        loop {
            let take_count = rand::random::<u32>() as usize % bytes.len();
            let taken = bytes.drain(..take_count).collect::<Vec<_>>();
            match parser.parse(&state, &taken) {
                Ok(result) => match result {
                    ParseStatus::Incomplete { new_state, .. } => {
                        state = new_state;
                    }
                    ParseStatus::Finished { result, remaining } => {
                        assert_eq!(result, random_number);
                        assert!(cap_string.as_bytes().starts_with(remaining));
                        break;
                    }
                },
                Err(_) => panic!("should parse correctly failed to parse {as_string:?}"),
            }
        }
    }
}
