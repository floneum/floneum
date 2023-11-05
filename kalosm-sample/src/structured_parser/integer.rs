use crate::{CreateParserState, ParseResult, Parser};
use std::ops::RangeInclusive;

/// A parser for an integer.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct IntegerParser {
    range: RangeInclusive<i64>,
}

impl IntegerParser {
    /// Create a new integer parser.
    pub fn new(range: RangeInclusive<i64>) -> Self {
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
    fn sign_valid(&self, positive: bool) -> bool {
        if positive {
            *self.range.start() >= 0
        } else {
            *self.range.end() <= 0
        }
    }

    fn is_number_valid(&self, value: i64) -> bool {
        self.range.contains(&value)
    }

    fn should_stop(&self, value: i64) -> bool {
        let after_next_digit = value * 10;
        after_next_digit > *self.range.end() || after_next_digit < *self.range.start()
    }

    fn could_number_become_valid(&self, value: i64) -> bool {
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
            for digit in 1..(digits + 1) {
                let selected_digit = value / (10_i64.pow(digits - digit)) % 10;
                let selected_start_digit = start_value / (10_i64.pow(start_digits - digit)) % 10;
                let selected_end_digit = end_value / (10_i64.pow(end_digits - digit)) % 10;
                if positive {
                    if selected_digit > selected_end_digit || selected_digit < selected_start_digit
                    {
                        return false;
                    }
                } else if selected_digit < selected_end_digit
                    || selected_digit > selected_start_digit
                {
                    return false;
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
    type Error = ();
    type Output = i64;
    type PartialState = IntegerParserState;

    fn parse<'a>(
        &self,
        state: &IntegerParserState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        let mut value = state.value;
        let mut positive = state.positive;
        let mut state = state.state;

        for index in 0..input.len() {
            let input_byte = input[index];
            let digit = match input_byte {
                b'0'..=b'9' => {
                    if (state == IntegerParserProgress::Initial
                        || state == IntegerParserProgress::AfterSign)
                        && input_byte == b'0'
                    {
                        return Err(()); // Leading zeros are not allowed
                    }
                    input_byte - b'0'
                }
                b'+' | b'-' => {
                    if state == IntegerParserProgress::Initial {
                        state = IntegerParserProgress::AfterSign;
                        positive = input_byte == b'+';
                        if !self.sign_valid(positive) {
                            return Err(());
                        }
                        continue;
                    } else {
                        return Err(());
                    }
                }
                _ => {
                    if state.is_after_digit() {
                        let result = value as i64 * if positive { 1 } else { -1 };
                        if self.is_number_valid(result) {
                            return Ok(ParseResult::Finished {
                                result,
                                remaining: &input[index..],
                            });
                        }
                        return Err(());
                    } else {
                        return Err(());
                    }
                }
            };

            state = IntegerParserProgress::AfterDigit;
            match value.checked_mul(10) {
                Some(v) => value = v + u64::from(digit),
                None => {
                    return Err(());
                }
            }

            if self.should_stop(value as i64 * if positive { 1 } else { -1 }) {
                return Ok(ParseResult::Finished {
                    result: value as i64 * if positive { 1 } else { -1 },
                    remaining: &input[index + 1..],
                });
            }

            if !self.could_number_become_valid(value as i64 * if positive { 1 } else { -1 }) {
                return Err(());
            }
        }

        Ok(ParseResult::Incomplete(IntegerParserState {
            state,
            value,
            positive,
        }))
    }
}

#[test]
fn integer_parser() {
    for _ in 0..100 {
        let random_number = rand::random::<i64>();

        let parser = IntegerParser {
            range: random_number - rand::random::<u8>() as i64
                ..=random_number + rand::random::<u8>() as i64,
        };
        let mut state = IntegerParserState::default();

        let mut as_string = random_number.to_string();
        let cap_string = rand::random::<char>().to_string();
        as_string += &cap_string;
        let mut bytes = as_string.as_bytes().to_vec();
        loop {
            let take_count = rand::random::<usize>() % bytes.len();
            let taken = bytes.drain(..take_count).collect::<Vec<_>>();
            match parser.parse(&state, &taken) {
                Ok(result) => match result {
                    ParseResult::Incomplete(new_state) => {
                        state = new_state;
                    }
                    ParseResult::Finished { result, remaining } => {
                        assert_eq!(result, random_number);
                        assert!(cap_string.as_bytes().starts_with(remaining));
                        break;
                    }
                },
                Err(_) => panic!("should parse correctly failed to parse {:?}", as_string),
            }
        }
    }
}
