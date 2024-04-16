use crate::{CreateParserState, ParseStatus, Parser};
use std::ops::RangeInclusive;

#[derive(Debug, PartialEq, Eq, Default, Copy, Clone)]
enum FloatParserProgress {
    #[default]
    Initial,
    AfterSign,
    AfterDigit,
    AfterDecimalPoint {
        digits_after_decimal_point: u32,
    },
}

impl FloatParserProgress {
    fn is_after_digit(&self) -> bool {
        matches!(
            self,
            FloatParserProgress::AfterDigit | FloatParserProgress::AfterDecimalPoint { .. }
        )
    }
}

/// The state of an integer parser.
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct FloatParserState {
    state: FloatParserProgress,
    value: f64,
    positive: bool,
}

impl Default for FloatParserState {
    fn default() -> Self {
        Self {
            state: FloatParserProgress::Initial,
            value: 0.0,
            positive: true,
        }
    }
}

/// A parser for a float.
#[derive(Debug, PartialEq, Clone)]
pub struct FloatParser {
    range: RangeInclusive<f64>,
}

impl FloatParser {
    /// Create a new float parser.
    pub fn new(range: RangeInclusive<f64>) -> Self {
        if range.start() > range.end() {
            Self {
                range: *range.end()..=*range.start(),
            }
        } else {
            Self { range }
        }
    }
}

impl CreateParserState for FloatParser {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        FloatParserState::default()
    }
}

impl FloatParser {
    fn sign_valid(&self, positive: bool) -> bool {
        if positive {
            *self.range.start() >= 0.0
        } else {
            *self.range.end() <= 0.0
        }
    }

    fn is_number_valid(&self, value: f64) -> bool {
        self.range.contains(&value)
    }

    fn could_number_become_valid_before_decimal(
        &self,
        value: f64,
        state: FloatParserProgress,
    ) -> bool {
        if self.is_number_valid(value) {
            true
        } else {
            let num_with_extra_digit = value * 10.;
            if value < 0. {
                if *self.range.start() > num_with_extra_digit {
                    return false;
                }
            } else if *self.range.end() < num_with_extra_digit {
                return false;
            }
            let value_string = value.abs().to_string();
            let start_value_string = self.range.start().abs().to_string();
            let end_value_string = self.range.end().abs().to_string();
            match state {
                FloatParserProgress::AfterDigit | FloatParserProgress::AfterSign => {
                    // Check if the digits are within the range so far
                    let digits = value_string.chars();
                    let start_digits = start_value_string.chars();
                    let end_digits = end_value_string.chars();
                    for (digit, (start_digit, end_digit)) in
                        digits.zip(start_digits.zip(end_digits))
                    {
                        if digit < start_digit || digit > end_digit {
                            return false;
                        }
                    }
                }
                _ => {}
            }
            true
        }
    }

    fn could_number_become_valid_after_decimal(
        &self,
        value: f64,
        digits_after_decimal_point: u32,
    ) -> bool {
        let distance = if value < 0.0 {
            *self.range.start() - value
        } else {
            *self.range.end() - value
        };
        println!("Distance: {}", distance);

        distance < 10.0_f64.powi(-(digits_after_decimal_point as i32))
    }
}

/// An error that can occur while parsing a float literal when the number starts with a leading zero.
#[derive(Debug)]
pub struct LeadingZeroError;

impl std::fmt::Display for LeadingZeroError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Found leading zero. Leading zeros are not allowed when parsing a number"
        )
    }
}

impl std::error::Error for LeadingZeroError {}

/// An error that can occur while parsing a float literal when the number is out of range.
#[derive(Debug)]
pub struct OutOfRangeError;

impl std::fmt::Display for OutOfRangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Attempted to parse a number that was out of range")
    }
}

impl std::error::Error for OutOfRangeError {}

/// An error that can occur while parsing a float literal when the number contains a decimal point in the wrong place.
#[derive(Debug)]
pub struct InvalidDecimalLocation;

impl std::fmt::Display for InvalidDecimalLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Failed to parse a number with a decimal before the first digit or multiple decimals"
        )
    }
}

impl std::error::Error for InvalidDecimalLocation {}

/// An error that can occur while parsing a float literal when the number contains a sign in the wrong place.
#[derive(Debug)]
pub struct InvalidSignLocation;

impl std::fmt::Display for InvalidSignLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Failed to parse a number with a sign after the first character"
        )
    }
}

impl std::error::Error for InvalidSignLocation {}

/// An error that can occur while parsing a float literal when trying to parse a number with no characters.
#[derive(Debug)]
pub struct EmptyNumber;

impl std::fmt::Display for EmptyNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Failed to parse a number with no digits")
    }
}

impl std::error::Error for EmptyNumber {}

impl Parser for FloatParser {
    type Output = f64;
    type PartialState = FloatParserState;

    fn parse<'a>(
        &self,
        state: &FloatParserState,
        input: &'a [u8],
    ) -> crate::ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        let mut value = state.value;
        let mut positive = state.positive;
        let mut state = state.state;

        for index in 0..input.len() {
            let input_byte = input[index];
            let digit = match input_byte {
                b'0'..=b'9' => {
                    if (state == FloatParserProgress::Initial
                        || state == FloatParserProgress::AfterSign)
                        && input_byte == b'0'
                    {
                        crate::bail!(LeadingZeroError);
                    }
                    input_byte - b'0'
                }
                b'.' => {
                    let value_digits = value.abs().log10() + 1.;
                    let start_digits = self.range.start().abs().log10() + 1.;
                    let end_digits = self.range.end().abs().log10() + 1.;
                    if positive {
                        if value_digits > end_digits {
                            crate::bail!(OutOfRangeError);
                        }
                    } else if value_digits > start_digits {
                        crate::bail!(OutOfRangeError);
                    }
                    if state == FloatParserProgress::AfterDigit {
                        state = FloatParserProgress::AfterDecimalPoint {
                            digits_after_decimal_point: 0,
                        };
                        continue;
                    } else {
                        crate::bail!(InvalidDecimalLocation);
                    }
                }
                b'+' | b'-' => {
                    if state == FloatParserProgress::Initial {
                        state = FloatParserProgress::AfterSign;
                        positive = input_byte == b'+';

                        if !self.sign_valid(positive) {
                            crate::bail!(InvalidSignLocation);
                        }
                        continue;
                    } else {
                        crate::bail!(InvalidSignLocation);
                    }
                }
                _ => {
                    if state.is_after_digit() {
                        let result = value * if positive { 1.0 } else { -1.0 };
                        if self.is_number_valid(result) {
                            return Ok(ParseStatus::Finished {
                                result,
                                remaining: &input[index..],
                            });
                        }
                        return Ok(ParseStatus::Finished {
                            result,
                            remaining: &input[index..],
                        });
                    } else {
                        crate::bail!(EmptyNumber)
                    }
                }
            };

            match &mut state {
                FloatParserProgress::Initial => {
                    state = FloatParserProgress::AfterDigit;
                    value = f64::from(digit);
                }
                FloatParserProgress::AfterSign => {
                    state = FloatParserProgress::AfterDigit;
                    value = f64::from(digit);
                }
                FloatParserProgress::AfterDigit => {
                    value = value * 10.0 + f64::from(digit);

                    if !self.could_number_become_valid_before_decimal(
                        value * if positive { 1.0 } else { -1.0 },
                        FloatParserProgress::AfterDigit,
                    ) {
                        crate::bail!(OutOfRangeError);
                    }
                }
                FloatParserProgress::AfterDecimalPoint {
                    digits_after_decimal_point,
                } => {
                    value +=
                        f64::from(digit) / 10.0_f64.powi(*digits_after_decimal_point as i32 + 1);
                    *digits_after_decimal_point += 1;

                    let signed_value = value * if positive { 1.0 } else { -1.0 };
                    if !self.range.contains(&signed_value)
                        && !self.could_number_become_valid_after_decimal(
                            signed_value,
                            *digits_after_decimal_point,
                        )
                    {
                        crate::bail!(OutOfRangeError);
                    }
                }
            }
        }

        Ok(ParseStatus::Incomplete {
            new_state: FloatParserState {
                state,
                value,
                positive,
            },
            required_next: Default::default(),
        })
    }
}

#[test]
fn float_parser() {
    let parser = FloatParser {
        range: -100.0..=200.0,
    };
    let state = FloatParserState::default();
    assert_eq!(
        parser.parse(&state, b"123").unwrap(),
        ParseStatus::Incomplete {
            new_state: FloatParserState {
                state: FloatParserProgress::AfterDigit,
                value: 123.0,
                positive: true
            },
            required_next: Default::default()
        }
    );
    assert_eq!(
        parser.parse(&state, b"123.456").unwrap(),
        ParseStatus::Incomplete {
            new_state: FloatParserState {
                state: FloatParserProgress::AfterDecimalPoint {
                    digits_after_decimal_point: 3
                },
                value: 123.456,
                positive: true
            },
            required_next: Default::default()
        }
    );
    assert_eq!(
        parser
            .parse(
                &parser
                    .parse(&state, b"123.456")
                    .unwrap()
                    .unwrap_incomplete()
                    .0,
                b"789x"
            )
            .unwrap(),
        ParseStatus::Finished {
            result: 123.456789,
            remaining: b"x"
        }
    );
    assert_eq!(
        parser.parse(&state, b"123.456x").unwrap(),
        ParseStatus::Finished {
            result: 123.456,
            remaining: b"x"
        }
    );
    assert!(parser.parse(&state, b"abc").is_err());
}
