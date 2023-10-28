use std::ops::RangeInclusive;

/// A trait for a parser with a default state.
pub trait CreateParserState: Parser {
    /// Create the default state of the parser.
    fn create_parser_state(&self) -> <Self as Parser>::PartialState;
}

impl<P: CreateParserState> CreateParserState for &P {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        (*self).create_parser_state()
    }
}

impl<P: CreateParserState> CreateParserState for Box<P> {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        (**self).create_parser_state()
    }
}

/// An incremental parser for a structured input.
pub trait Parser {
    /// The error type of the parser.
    type Error;
    /// The output of the parser.
    type Output;
    /// The state of the parser.
    type PartialState;

    /// Parse the given input.
    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error>
    where
        Self: Sized;

    /// Parse this parser, or another other parser.
    fn or<V: Parser<Error = E, Output = O, PartialState = PA>, E, O, PA>(
        self,
        other: V,
    ) -> ChoiceParser<Self, V>
    where
        Self: Sized,
    {
        ChoiceParser {
            parser1: self,
            parser2: other,
        }
    }

    /// Parse this parser, then the other parser.
    fn then<V: Parser<Error = E, Output = O, PartialState = PA>, E, O, PA>(
        self,
        other: V,
    ) -> SequenceParser<Self, V>
    where
        Self: Sized,
    {
        SequenceParser {
            parser1: self,
            parser2: other,
        }
    }
}

impl<P: Parser> Parser for &P {
    type Error = P::Error;
    type Output = P::Output;
    type PartialState = P::PartialState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        (*self).parse(state, input)
    }
}

impl<P: Parser> Parser for Box<P> {
    type Error = P::Error;
    type Output = P::Output;
    type PartialState = P::PartialState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        let _self: &P = &*self;
        _self.parse(state, input)
    }
}

/// A parser for a choice between two parsers.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum OwnedParseResult<P, R> {
    /// The parser is incomplete.
    Incomplete(P),
    /// The parser is finished.
    Finished {
        /// The result of the parser.
        result: R,
        /// The remaining input.
        remaining: Vec<u8>,
    },
}

impl<P, R> From<ParseResult<'_, P, R>> for OwnedParseResult<P, R> {
    fn from(result: ParseResult<P, R>) -> Self {
        match result {
            ParseResult::Incomplete(parser) => OwnedParseResult::Incomplete(parser),
            ParseResult::Finished { result, remaining } => OwnedParseResult::Finished {
                result,
                remaining: remaining.to_vec(),
            },
        }
    }
}

/// The state of a parser.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum ParseResult<'a, P, R> {
    /// The parser is incomplete.
    Incomplete(P),
    /// The parser is finished.
    Finished {
        /// The result of the parser.
        result: R,
        /// The remaining input.
        remaining: &'a [u8],
    },
}

impl<'a, P, R> ParseResult<'a, P, R> {
    /// Take the remaining bytes from the parser.
    pub fn without_remaining(self) -> ParseResult<'static, P, R> {
        match self {
            ParseResult::Finished { result, .. } => ParseResult::Finished {
                result,
                remaining: &[],
            },
            ParseResult::Incomplete(parser) => ParseResult::Incomplete(parser),
        }
    }

    /// Unwrap the parser to a finished result.
    pub fn unwrap_finished(self) -> R {
        match self {
            ParseResult::Finished { result, .. } => result,
            ParseResult::Incomplete(_) => {
                panic!("called `ParseResult::unwrap_finished()` on an `Incomplete` value")
            }
        }
    }

    /// Unwrap the parser to an incomplete result.
    pub fn unwrap_incomplete(self) -> P {
        match self {
            ParseResult::Finished { .. } => {
                panic!("called `ParseResult::unwrap_incomplete()` on a `Finished` value")
            }
            ParseResult::Incomplete(parser) => parser,
        }
    }
}

/// A parser for a literal.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct LiteralParser<S: AsRef<str>> {
    literal: S,
}

impl<S: AsRef<str>> CreateParserState for LiteralParser<S> {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        LiteralParserOffset::default()
    }
}

impl<S: AsRef<str>> From<S> for LiteralParser<S> {
    fn from(literal: S) -> Self {
        Self { literal }
    }
}

/// The state of a literal parser.
#[derive(Default, Debug, PartialEq, Eq, Copy, Clone)]
pub struct LiteralParserOffset {
    offset: usize,
}

impl<S: AsRef<str>> Parser for LiteralParser<S> {
    type Error = ();
    type Output = ();
    type PartialState = LiteralParserOffset;

    fn parse<'a>(
        &self,
        state: &LiteralParserOffset,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        let mut bytes_consumed = 0;

        for (input_byte, literal_byte) in input
            .iter()
            .zip(self.literal.as_ref().as_bytes()[state.offset..].iter())
        {
            if input_byte != literal_byte {
                return Err(());
            }
            bytes_consumed += 1;
        }

        if state.offset + bytes_consumed == self.literal.as_ref().len() {
            Ok(ParseResult::Finished {
                result: (),
                remaining: &input[bytes_consumed..],
            })
        } else {
            Ok(ParseResult::Incomplete(LiteralParserOffset {
                offset: state.offset + bytes_consumed,
            }))
        }
    }
}

#[test]
fn literal_parser() {
    let parser = LiteralParser {
        literal: "Hello, world!",
    };
    let state = LiteralParserOffset { offset: 0 };
    assert_eq!(
        parser.parse(&state, b"Hello, world!"),
        Ok(ParseResult::Finished {
            result: (),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"Hello, "),
        Ok(ParseResult::Incomplete(LiteralParserOffset { offset: 7 }))
    );
    assert_eq!(
        parser.parse(
            &parser
                .parse(&state, b"Hello, ")
                .unwrap()
                .unwrap_incomplete(),
            b"world!"
        ),
        Ok(ParseResult::Finished {
            result: (),
            remaining: &[]
        })
    );
    assert_eq!(parser.parse(&state, b"Goodbye, world!"), Err(()));
}

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
            } else {
                if value * 10 < start_value {
                    return false;
                }
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
                } else {
                    if selected_digit < selected_end_digit || selected_digit > selected_start_digit
                    {
                        return false;
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
                b'0'..=b'9' => input_byte - b'0',
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
            } else {
                if *self.range.end() < num_with_extra_digit {
                    return false;
                }
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

        if distance < 10.0_f64.powi(-(digits_after_decimal_point as i32)) {
            true
        } else {
            false
        }
    }
}

impl Parser for FloatParser {
    type Error = ();
    type Output = f64;
    type PartialState = FloatParserState;

    fn parse<'a>(
        &self,
        state: &FloatParserState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        let mut value = state.value;
        let mut positive = state.positive;
        let mut state = state.state;

        for index in 0..input.len() {
            let input_byte = input[index];
            let digit = match input_byte {
                b'0'..=b'9' => input_byte - b'0',
                b'.' => {
                    let value_digits = value.abs().log10() + 1.;
                    let start_digits = self.range.start().abs().log10() + 1.;
                    let end_digits = self.range.end().abs().log10() + 1.;
                    if positive {
                        if value_digits > end_digits {
                            return Err(());
                        }
                    } else {
                        if value_digits > start_digits {
                            return Err(());
                        }
                    }
                    if state == FloatParserProgress::AfterDigit {
                        state = FloatParserProgress::AfterDecimalPoint {
                            digits_after_decimal_point: 0,
                        };
                        continue;
                    } else {
                        return Err(());
                    }
                }
                b'+' | b'-' => {
                    if state == FloatParserProgress::Initial {
                        state = FloatParserProgress::AfterSign;
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
                        let result = value * if positive { 1.0 } else { -1.0 };
                        if self.is_number_valid(result) {
                            return Ok(ParseResult::Finished {
                                result,
                                remaining: &input[index..],
                            });
                        }
                        return Ok(ParseResult::Finished {
                            result,
                            remaining: &input[index..],
                        });
                    } else {
                        return Err(());
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
                        return Err(());
                    }
                }
                FloatParserProgress::AfterDecimalPoint {
                    digits_after_decimal_point,
                } => {
                    value = value
                        + f64::from(digit) / 10.0_f64.powi(*digits_after_decimal_point as i32 + 1);
                    *digits_after_decimal_point += 1;

                    if !self.could_number_become_valid_after_decimal(
                        value * if positive { 1.0 } else { -1.0 },
                        *digits_after_decimal_point,
                    ) {
                        return Err(());
                    }
                }
            }
        }

        Ok(ParseResult::Incomplete(FloatParserState {
            state,
            value,
            positive,
        }))
    }
}

#[test]
fn float_parser() {
    let parser = FloatParser {
        range: -100.0..=100.0,
    };
    let state = FloatParserState::default();
    assert_eq!(
        parser.parse(&state, b"123"),
        Ok(ParseResult::Incomplete(FloatParserState {
            state: FloatParserProgress::AfterDigit,
            value: 123.0,
            positive: true
        }))
    );
    assert_eq!(
        parser.parse(&state, b"123.456"),
        Ok(ParseResult::Incomplete(FloatParserState {
            state: FloatParserProgress::AfterDecimalPoint {
                digits_after_decimal_point: 3
            },
            value: 123.456,
            positive: true
        }))
    );
    assert_eq!(
        parser.parse(
            &parser
                .parse(&state, b"123.456")
                .unwrap()
                .unwrap_incomplete(),
            b"789x"
        ),
        Ok(ParseResult::Finished {
            result: 123.456789,
            remaining: b"x"
        })
    );
    assert_eq!(
        parser.parse(&state, b"123.456x"),
        Ok(ParseResult::Finished {
            result: 123.456,
            remaining: b"x"
        })
    );
    assert_eq!(parser.parse(&state, b"abc"), Err(()));
}

/// State of a sequence parser.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum SequenceParserState<P1, P2, O1> {
    /// The first parser is incomplete.
    FirstParser(P1),
    /// The first parser is finished, and the second parser is incomplete.
    SecondParser(P2, O1),
}

impl<P1: Default, P2, O1> Default for SequenceParserState<P1, P2, O1> {
    fn default() -> Self {
        SequenceParserState::FirstParser(Default::default())
    }
}

impl<
        E,
        O1: Clone,
        O2,
        PA1,
        PA2,
        P1: Parser<Error = E, Output = O1, PartialState = PA1> + CreateParserState,
        P2: Parser<Error = E, Output = O2, PartialState = PA2> + CreateParserState,
    > CreateParserState for SequenceParser<P1, P2>
{
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        SequenceParserState::FirstParser(self.parser1.create_parser_state())
    }
}

/// A parser for a sequence of two parsers.
#[derive(Default, Debug, PartialEq, Eq, Copy, Clone)]
pub struct SequenceParser<P1, P2> {
    parser1: P1,
    parser2: P2,
}

impl<
        E,
        O1: Clone,
        O2,
        PA1,
        PA2,
        P1: Parser<Error = E, Output = O1, PartialState = PA1>,
        P2: Parser<Error = E, Output = O2, PartialState = PA2> + CreateParserState,
    > Parser for SequenceParser<P1, P2>
{
    type Error = E;
    type Output = (O1, O2);
    type PartialState = SequenceParserState<PA1, PA2, O1>;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        match state {
            SequenceParserState::FirstParser(p1) => {
                let result = self.parser1.parse(p1, input)?;
                match result {
                    ParseResult::Finished {
                        result: o1,
                        remaining,
                    } => {
                        let second_parser_state = self.parser2.create_parser_state();
                        let result = self.parser2.parse(&second_parser_state, remaining)?;
                        match result {
                            ParseResult::Finished { result, remaining } => {
                                Ok(ParseResult::Finished {
                                    result: (o1, result),
                                    remaining,
                                })
                            }
                            ParseResult::Incomplete(p2) => {
                                let new_state = SequenceParserState::SecondParser(p2, o1);
                                Ok(ParseResult::Incomplete(new_state))
                            }
                        }
                    }
                    ParseResult::Incomplete(p1) => {
                        let new_state = SequenceParserState::FirstParser(p1);
                        Ok(ParseResult::Incomplete(new_state))
                    }
                }
            }
            SequenceParserState::SecondParser(p2, o1) => {
                let result = self.parser2.parse(p2, input)?;
                match result {
                    ParseResult::Finished { result, remaining } => Ok(ParseResult::Finished {
                        result: (o1.clone(), result),
                        remaining,
                    }),
                    ParseResult::Incomplete(p2) => {
                        let new_state = SequenceParserState::SecondParser(p2, o1.clone());
                        Ok(ParseResult::Incomplete(new_state))
                    }
                }
            }
        }
    }
}

#[test]
fn sequence_parser() {
    let parser = SequenceParser {
        parser1: LiteralParser { literal: "Hello, " },
        parser2: LiteralParser { literal: "world!" },
    };
    let state = SequenceParserState::FirstParser(LiteralParserOffset::default());
    assert_eq!(
        parser.parse(&state, b"Hello, world!"),
        Ok(ParseResult::Finished {
            result: ((), ()),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"Hello, "),
        Ok(ParseResult::Incomplete(SequenceParserState::SecondParser(
            LiteralParserOffset { offset: 0 },
            ()
        )))
    );
    assert_eq!(
        parser.parse(
            &parser
                .parse(&state, b"Hello, ")
                .unwrap()
                .unwrap_incomplete(),
            b"world!"
        ),
        Ok(ParseResult::Finished {
            result: ((), ()),
            remaining: &[]
        })
    );
    assert_eq!(parser.parse(&state, b"Goodbye, world!"), Err(()));
}

/// State of a choice parser.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct ChoiceParserState<P1, P2, E> {
    state1: Result<P1, E>,
    state2: Result<P2, E>,
}

impl<P1: Default, P2: Default, E> Default for ChoiceParserState<P1, P2, E> {
    fn default() -> Self {
        ChoiceParserState {
            state1: Ok(Default::default()),
            state2: Ok(Default::default()),
        }
    }
}

/// A parser for a choice of two parsers.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Default)]
pub struct ChoiceParser<P1, P2> {
    parser1: P1,
    parser2: P2,
}

impl<
        E: Clone,
        O1,
        O2,
        PA1,
        PA2,
        P1: Parser<Error = E, Output = O1, PartialState = PA1> + CreateParserState,
        P2: Parser<Error = E, Output = O2, PartialState = PA2> + CreateParserState,
    > CreateParserState for ChoiceParser<P1, P2>
{
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        ChoiceParserState {
            state1: Ok(self.parser1.create_parser_state()),
            state2: Ok(self.parser2.create_parser_state()),
        }
    }
}

/// A value that can be one of two types.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Either<L, R> {
    /// The value is the left type.
    Left(L),
    /// The value is the right type.
    Right(R),
}

impl<
        E: Clone,
        O1,
        O2,
        PA1,
        PA2,
        P1: Parser<Error = E, Output = O1, PartialState = PA1>,
        P2: Parser<Error = E, Output = O2, PartialState = PA2>,
    > Parser for ChoiceParser<P1, P2>
{
    type Error = E;
    type Output = Either<O1, O2>;
    type PartialState = ChoiceParserState<PA1, PA2, E>;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        match (&state.state1, &state.state2) {
            (Ok(p1), Ok(p2)) => {
                match (self.parser1.parse(p1, input), self.parser2.parse(p2, input)) {
                    // If one parser finishes, we return the result of that parser
                    (Ok(ParseResult::Finished { result, remaining }), _) => {
                        Ok(ParseResult::Finished {
                            result: Either::Left(result),
                            remaining,
                        })
                    }
                    (_, Ok(ParseResult::Finished { result, remaining })) => {
                        Ok(ParseResult::Finished {
                            result: Either::Right(result),
                            remaining,
                        })
                    }
                    // If either parser is incomplete, we return the incomplete state
                    (Ok(ParseResult::Incomplete(p1)), Ok(ParseResult::Incomplete(p2))) => {
                        let new_state = ChoiceParserState {
                            state1: Ok(p1),
                            state2: Ok(p2),
                        };
                        Ok(ParseResult::Incomplete(new_state))
                    }
                    (Ok(ParseResult::Incomplete(p1)), Err(err2)) => {
                        let new_state = ChoiceParserState {
                            state1: Ok(p1),
                            state2: Err(err2),
                        };
                        Ok(ParseResult::Incomplete(new_state))
                    }
                    (Err(err1), Ok(ParseResult::Incomplete(p2))) => {
                        let new_state = ChoiceParserState {
                            state1: Err(err1),
                            state2: Ok(p2),
                        };
                        Ok(ParseResult::Incomplete(new_state))
                    }

                    // If both parsers fail, we return the error from the first parser
                    (Err(err1), Err(_)) => Err(err1),
                }
            }
            (Ok(p1), Err(err2)) => {
                let result = self.parser1.parse(p1, input)?;
                match result {
                    ParseResult::Finished { result, remaining } => Ok(ParseResult::Finished {
                        result: Either::Left(result),
                        remaining,
                    }),
                    ParseResult::Incomplete(p1) => {
                        let new_state = ChoiceParserState {
                            state1: Ok(p1),
                            state2: Err(err2.clone()),
                        };
                        Ok(ParseResult::Incomplete(new_state))
                    }
                }
            }
            (Err(err1), Ok(p2)) => {
                let result = self.parser2.parse(p2, input)?;
                match result {
                    ParseResult::Finished { result, remaining } => Ok(ParseResult::Finished {
                        result: Either::Right(result),
                        remaining,
                    }),
                    ParseResult::Incomplete(p2) => {
                        let new_state = ChoiceParserState {
                            state1: Err(err1.clone()),
                            state2: Ok(p2),
                        };
                        Ok(ParseResult::Incomplete(new_state))
                    }
                }
            }
            (Err(_), Err(_)) => {
                unreachable!()
            }
        }
    }
}

#[test]
fn choice_parser() {
    let parser = ChoiceParser {
        parser1: LiteralParser { literal: "Hello, " },
        parser2: LiteralParser { literal: "world!" },
    };
    let state = ChoiceParserState::default();
    assert_eq!(
        parser.parse(&state, b"Hello, "),
        Ok(ParseResult::Finished {
            result: Either::Left(()),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"Hello, "),
        Ok(ParseResult::Finished {
            result: Either::Left(()),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"world!"),
        Ok(ParseResult::Finished {
            result: Either::Right(()),
            remaining: &[]
        })
    );
    assert_eq!(parser.parse(&state, b"Goodbye, world!"), Err(()));

    let parser = ChoiceParser {
        parser1: LiteralParser {
            literal: "This isn't a test",
        },
        parser2: LiteralParser {
            literal: "This is a test",
        },
    };
    let state = ChoiceParserState::default();
    assert_eq!(
        parser.parse(&state, b"This isn"),
        Ok(ParseResult::Incomplete(ChoiceParserState {
            state1: Ok(LiteralParserOffset { offset: 8 }),
            state2: Err(()),
        }))
    );
}

/// A validator for a string
#[derive(Debug, Clone)]
pub enum StructureParser {
    /// A literal string
    Literal(String),
    /// A number
    Num {
        /// The minimum value of the number
        min: f64,
        /// The maximum value of the number
        max: f64,
        /// If the number must be an integer
        integer: bool,
    },
    /// Either the first or the second parser
    Either {
        /// The first parser
        first: Box<StructureParser>,
        /// The second parser
        second: Box<StructureParser>,
    },
    /// The first parser, then the second parser
    Then {
        /// The first parser
        first: Box<StructureParser>,
        /// The second parser
        second: Box<StructureParser>,
    },
}

/// The state of a structure parser.
#[allow(missing_docs)]
#[derive(Debug, PartialEq, Clone)]
pub enum StructureParserState {
    Literal(LiteralParserOffset),
    NumInt(IntegerParserState),
    Num(FloatParserState),
    Either(ChoiceParserState<Box<StructureParserState>, Box<StructureParserState>, ()>),
    Then(SequenceParserState<Box<StructureParserState>, Box<StructureParserState>, ()>),
}

impl CreateParserState for StructureParser {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        match self {
            StructureParser::Literal(literal) => {
                StructureParserState::Literal(LiteralParser::from(literal).create_parser_state())
            }
            StructureParser::Num { min, max, integer } => {
                if *integer {
                    StructureParserState::NumInt(
                        IntegerParser {
                            range: *min as i64..=*max as i64,
                        }
                        .create_parser_state(),
                    )
                } else {
                    StructureParserState::Num(
                        FloatParser { range: *min..=*max }.create_parser_state(),
                    )
                }
            }
            StructureParser::Either { first, second } => {
                StructureParserState::Either(ChoiceParserState {
                    state1: Ok(Box::new(first.create_parser_state())),
                    state2: Ok(Box::new(second.create_parser_state())),
                })
            }
            StructureParser::Then { first, .. } => StructureParserState::Then(
                SequenceParserState::FirstParser(Box::new(first.create_parser_state())),
            ),
        }
    }
}

impl Parser for StructureParser {
    type Error = ();
    type Output = ();
    type PartialState = StructureParserState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error>
    where
        Self: Sized,
    {
        match (self, state) {
            (StructureParser::Literal(lit_parser), StructureParserState::Literal(state)) => {
                LiteralParser::from(lit_parser)
                    .parse(state, input)
                    .map(|result| match result {
                        ParseResult::Finished { result, remaining } => {
                            ParseResult::Finished { result, remaining }
                        }
                        ParseResult::Incomplete(parser) => {
                            ParseResult::Incomplete(StructureParserState::Literal(parser))
                        }
                    })
            }
            (
                StructureParser::Num {
                    min,
                    max,
                    integer: false,
                },
                StructureParserState::Num(state),
            ) => {
                FloatParser { range: *min..=*max }
                    .parse(state, input)
                    .map(|result| match result {
                        ParseResult::Finished { remaining, .. } => ParseResult::Finished {
                            result: (),
                            remaining,
                        },
                        ParseResult::Incomplete(parser) => {
                            ParseResult::Incomplete(StructureParserState::Num(parser))
                        }
                    })
            }
            (
                StructureParser::Num {
                    min,
                    max,
                    integer: true,
                },
                StructureParserState::NumInt(int),
            ) => IntegerParser {
                range: *min as i64..=*max as i64,
            }
            .parse(int, input)
            .map(|result| match result {
                ParseResult::Finished { remaining, .. } => ParseResult::Finished {
                    result: (),
                    remaining,
                },
                ParseResult::Incomplete(parser) => {
                    ParseResult::Incomplete(StructureParserState::NumInt(parser))
                }
            }),
            (StructureParser::Either { first, second }, StructureParserState::Either(state)) => {
                let state = ChoiceParserState {
                    state1: match &state.state1 {
                        Ok(state) => Ok((**state).clone()),
                        Err(()) => Err(()),
                    },
                    state2: match &state.state2 {
                        Ok(state) => Ok((**state).clone()),
                        Err(()) => Err(()),
                    },
                };
                let parser = ChoiceParser {
                    parser1: first.clone(),
                    parser2: second.clone(),
                };
                match parser.parse(&state, input) {
                    Ok(ParseResult::Incomplete(state)) => Ok(ParseResult::Incomplete(
                        StructureParserState::Either(ChoiceParserState {
                            state1: state.state1.map(Box::new),
                            state2: state.state2.map(Box::new),
                        }),
                    )),
                    Ok(ParseResult::Finished { remaining, .. }) => Ok(ParseResult::Finished {
                        result: (),
                        remaining,
                    }),
                    Err(_) => Err(()),
                }
            }
            (StructureParser::Then { first, second }, StructureParserState::Then(state)) => {
                let state = SequenceParserState::FirstParser(match &state {
                    SequenceParserState::FirstParser(state) => (**state).clone(),
                    SequenceParserState::SecondParser(state, _) => (**state).clone(),
                });
                let parser = SequenceParser {
                    parser1: first.clone(),
                    parser2: second.clone(),
                };
                match parser.parse(&state, input) {
                    Ok(ParseResult::Incomplete(state)) => Ok(ParseResult::Incomplete(
                        StructureParserState::Then(match state {
                            SequenceParserState::FirstParser(state) => {
                                SequenceParserState::FirstParser(Box::new(state))
                            }
                            SequenceParserState::SecondParser(state, _) => {
                                SequenceParserState::SecondParser(Box::new(state), ())
                            }
                        }),
                    )),
                    Ok(ParseResult::Finished { remaining, .. }) => Ok(ParseResult::Finished {
                        result: (),
                        remaining,
                    }),
                    Err(_) => Err(()),
                }
            }
            _ => unreachable!(),
        }
    }
}
