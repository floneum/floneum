use std::iter::Peekable;
use std::str::Chars;
use std::{cell::RefCell, ops::Deref};

/// A validator for a string
#[derive(Debug, Clone)]
pub enum StructureParser {
    /// A literal string
    Literal(String),
    /// A sequence of items separated by a separator
    Sequence {
        /// The item to parse
        item: Box<StructureParser>,
        /// The separator to parse
        separator: Box<StructureParser>,
        /// The minimum number of items to parse
        min_len: u64,
        /// The maximum number of items to parse
        max_len: u64,
    },
    /// A number
    Num {
        /// The minimum value of the number
        min: f64,
        /// The maximum value of the number
        max: f64,
        /// If the number must be an integer
        integer: bool,
    },
    /// A string
    String {
        /// The minimum length of the string
        min_len: u64,
        /// The maximum length of the string
        max_len: u64,
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

impl Validate for StructureParser {
    fn validate<'a>(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        match self {
            StructureParser::Literal(text) => text.validate(tokens),
            StructureParser::Sequence {
                item,
                separator,
                min_len,
                max_len,
            } => {
                let parse_sequence = Separated::new(
                    item.as_ref(),
                    separator.as_ref(),
                    *min_len as usize,
                    *max_len as usize,
                );

                parse_sequence.validate(tokens)
            }
            StructureParser::Num { min, max, integer } => {
                let parse_int = ValidateInt {
                    min: *min,
                    max: *max,
                    integer: *integer,
                };
                parse_int.validate(tokens)
            }
            StructureParser::String { min_len, max_len } => {
                let parse_string = ValidateString(*min_len, *max_len);
                parse_string.validate(tokens)
            }
            StructureParser::Either { first, second } => {
                first.deref().or(second.deref()).validate(tokens)
            }
            StructureParser::Then { first, second } => {
                first.deref().then(second.deref()).validate(tokens)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ValidateInt {
    min: f64,
    max: f64,
    integer: bool,
}

impl Validate for ValidateInt {
    fn validate<'a>(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        let ValidateInt { min, max, integer } = *self;
        let max_negative = max < 0.;
        let mut iter = tokens.iter();
        let mut current_number = 0.;
        let mut decimal_place = 0.0;

        let is_negative = iter.peek().copied() == Some('-');
        if is_negative {
            let _ = iter.next();
        }
        if !is_negative && max_negative {
            return ParseStatus::Invalid;
        }

        fn is_invalid(max: f64, min: f64, real_number: f64, decimal_place: f64) -> bool {
            // check if we've gone over the max and more digits would not shift the number enough
            if real_number > max && real_number - max > decimal_place {
                return true;
            }
            // check if we've gone under the min and more digits would not shift the number enough
            if real_number < min && min - real_number > decimal_place {
                return true;
            }
            false
        }

        let mut has_decimal = false;
        let mut has_digits = false;

        while let Some(c) = iter.peek() {
            let real_number = if is_negative {
                -current_number
            } else {
                current_number
            };
            match c {
                '0'..='9' => {
                    has_digits = true;
                    let digit = c.to_digit(10).unwrap() as i64;
                    if has_decimal {
                        current_number += digit as f64 * decimal_place;
                        decimal_place *= 0.1;
                    } else {
                        current_number = current_number * 10. + digit as f64;
                    }
                    let real_number = if is_negative {
                        -current_number
                    } else {
                        current_number
                    };
                    if is_invalid(max, min, real_number, decimal_place) {
                        return ParseStatus::Invalid;
                    }
                }
                '.' => {
                    if integer {
                        return ParseStatus::Invalid;
                    }
                    if has_decimal {
                        if has_digits {
                            if real_number > max || real_number < min {
                                return ParseStatus::Invalid;
                            }
                            return ParseStatus::Complete(Some(iter.into()));
                        } else {
                            return ParseStatus::Invalid;
                        }
                    }
                    has_decimal = true;
                    decimal_place = 0.1;
                }
                _ => {
                    if has_digits {
                        if real_number > max || real_number < min {
                            return ParseStatus::Invalid;
                        }
                        return ParseStatus::Complete(Some(iter.into()));
                    } else {
                        return ParseStatus::Invalid;
                    }
                }
            }
            let _ = iter.next();
        }

        ParseStatus::Incomplete {
            required_next: None,
        }
    }
}

#[test]
fn test_parse_num() {
    let tokens = ParseStream::new("-1234 ");
    assert!(ValidateInt {
        min: -2000.,
        max: 2000.,
        integer: true,
    }
    .validate(tokens)
    .is_complete());

    let tokens = ParseStream::new("1234hello");
    assert!(ValidateInt {
        min: -2000.,
        max: 2000.,
        integer: true,
    }
    .validate(tokens)
    .is_complete());

    let tokens = ParseStream::new("1234.0 ");
    assert!(ValidateInt {
        min: -2000.,
        max: 2000.,
        integer: false,
    }
    .validate(tokens)
    .is_complete());

    let tokens = ParseStream::new("1234.0.0");
    assert!(ValidateInt {
        min: -2000.,
        max: 2000.,
        integer: false,
    }
    .validate(tokens)
    .is_complete());
}

struct ValidateString(u64, u64);

impl Validate for ValidateString {
    fn validate<'a>(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        let min_len = self.0;
        let max_len = self.1;
        let mut iter = tokens.iter();
        let mut escape = false;

        if iter.next() != Some('"') {
            return ParseStatus::Invalid;
        }
        let mut string_length = 0;

        while let Some(c) = iter.peek() {
            match c {
                '\\' => {
                    if string_length >= max_len {
                        return ParseStatus::Invalid;
                    }
                    if escape {
                        string_length += 1;
                    }
                    escape = !escape;
                }
                '"' => {
                    if !escape {
                        if string_length < min_len {
                            return ParseStatus::Invalid;
                        }
                        let _ = iter.next();
                        return ParseStatus::Complete(iter.peek().is_some().then(|| iter.into()));
                    }
                    string_length += 1;
                    escape = false;
                }
                _ => {
                    string_length += 1;
                    escape = false;
                }
            }
            if string_length > max_len {
                return ParseStatus::Invalid;
            }
            let _ = iter.next();
        }

        ParseStatus::Incomplete {
            required_next: None,
        }
    }
}

#[test]
fn test_validate_string() {
    let tokens = ParseStream::new("\"hello\"");
    assert!(ValidateString(5, 5).validate(tokens).is_complete());

    let tokens = ParseStream::new("\"hello world");
    assert!(ValidateString(5, 50).validate(tokens).is_incomplete());

    let tokens = ParseStream::new("hello\"");
    assert!(ValidateString(5, 5).validate(tokens).is_invalid());
    let tokens = ParseStream::new("\"\"");
    assert!(StructureParser::String {
        min_len: 1,
        max_len: 5,
    }
    .validate(tokens)
    .is_invalid());
    let tokens = ParseStream::new("\"光");
    assert!(StructureParser::String {
        min_len: 1,
        max_len: 5,
    }
    .validate(tokens)
    .is_incomplete());
    let tokens = ParseStream::new("\"hello ");
    assert!(StructureParser::String {
        min_len: 1,
        max_len: 5,
    }
    .validate(tokens)
    .is_invalid());
    let tokens = ParseStream::new("\"\ndef!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    assert!(StructureParser::String {
        min_len: 1,
        max_len: 5,
    }
    .validate(tokens)
    .is_invalid());

    // Test escaping
    let stream = ParseStream::new("\"hello \\\"world\\\"\"");
    assert!(ValidateString(14, 14).validate(stream.clone()).is_invalid());
    assert!(ValidateString(13, 13)
        .validate(stream.clone())
        .is_complete());
    assert!(ValidateString(12, 12).validate(stream).is_invalid());
    let stream = ParseStream::new("\"hello \\");
    assert!(ValidateString(6, 6).validate(stream.clone()).is_invalid());
    assert!(ValidateString(7, 7)
        .validate(stream.clone())
        .is_incomplete());
}

/// The status of a the parser.
#[derive(Debug)]
pub enum ParseStatus<'a> {
    /// The parser is incomplete, but valid so far
    Incomplete {
        /// The next token that is required to complete the parser.
        required_next: Option<String>,
    },
    /// The parser is complete with the given tokens left over.
    Complete(Option<ParseStream<'a>>),
    /// The parser is invalid.
    Invalid,
}

#[allow(unused)]
impl ParseStatus<'_> {
    /// Check if this status is complete.
    pub fn is_complete(&self) -> bool {
        matches!(self, ParseStatus::Complete(_))
    }

    /// Check if this status is invalid.
    pub fn is_invalid(&self) -> bool {
        matches!(self, ParseStatus::Invalid)
    }

    /// Check if this status is incomplete.
    pub fn is_incomplete(&self) -> bool {
        matches!(self, ParseStatus::Incomplete { .. })
    }
}

/// A stream of tokens that can be parsed.
#[derive(Debug, Clone)]
pub struct ParseStream<'a> {
    chars: Peekable<Chars<'a>>,
}

impl<'a> From<Peekable<Chars<'a>>> for ParseStream<'a> {
    fn from(chars: Peekable<Chars<'a>>) -> Self {
        Self { chars }
    }
}

impl<'a> ParseStream<'a> {
    /// Create a new `ParseStream` from a string.
    pub fn new(token: &'a str) -> Self {
        Self {
            chars: token.chars().peekable(),
        }
    }

    /// Get an iterator over the remaining characters.
    pub fn iter(&self) -> Peekable<Chars<'a>> {
        self.chars.clone()
    }

    /// Check if this stream is empty.
    pub fn is_empty(&mut self) -> bool {
        self.chars.peek().is_none()
    }
}

/// A struct that checks if a set of tokens can be parsed.
pub trait Validate {
    /// Check if this parser can parse the given tokens.
    fn validate<'a>(&self, tokens: ParseStream<'a>) -> ParseStatus<'a>;

    /// Parse this parser, or another other parser.
    fn or<V: Validate>(self, other: V) -> Or<Self, V>
    where
        Self: Sized,
    {
        Or(self, other)
    }

    /// Parse this parser, then the other parser.
    fn then<V: Validate>(self, other: V) -> Then<Self, V>
    where
        Self: Sized,
    {
        Then(self, other)
    }

    /// Erase the type of this parser and return a boxed dynamic parser.
    fn boxed<'a>(self) -> BoxedValidate<'a>
    where
        Self: Sized + 'a,
    {
        BoxedValidate(Box::new(self))
    }
}

/// A boxed dynamic Validater.
pub struct BoxedValidate<'a>(pub(crate) Box<dyn Validate + 'a>);

impl<'j> Validate for BoxedValidate<'j> {
    fn validate<'a>(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        self.0.validate(tokens)
    }
}

impl<V: Validate> Validate for &V {
    fn validate<'a>(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        (*self).validate(tokens)
    }
}

struct Anonymous<F>(RefCell<F>);

#[allow(unused)]
impl<'a, F> Anonymous<F>
where
    F: FnMut(ParseStream<'a>) -> ParseStatus<'a>,
{
    fn new(f: F) -> Self {
        Self(RefCell::new(f))
    }
}

impl<F> Validate for Anonymous<F>
where
    F: for<'a> FnMut(ParseStream<'a>) -> ParseStatus<'a>,
{
    fn validate<'a>(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        (self.0.borrow_mut())(tokens)
    }
}

/// A parser that will parse the first parser, then the second parser.
pub struct Then<A: Validate, B: Validate>(A, B);

impl<A: Validate, B: Validate> Validate for Then<A, B> {
    fn validate<'a>(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        match self.0.validate(tokens) {
            ParseStatus::Complete(Some(tokens)) => self.1.validate(tokens),
            ParseStatus::Complete(None) => ParseStatus::Incomplete {
                required_next: None,
            },
            ParseStatus::Invalid => ParseStatus::Invalid,
            ParseStatus::Incomplete { required_next } => ParseStatus::Incomplete { required_next },
        }
    }
}

/// A parser that will parse either the first or the second parser.
pub struct Or<A: Validate, B: Validate>(A, B);

impl<A: Validate, B: Validate> Validate for Or<A, B> {
    fn validate<'a>(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        match self.0.validate(tokens.clone()) {
            ParseStatus::Complete(tokens) => ParseStatus::Complete(tokens),
            ParseStatus::Invalid => self.1.validate(tokens),
            ParseStatus::Incomplete { required_next } => match self.1.validate(tokens) {
                ParseStatus::Invalid => ParseStatus::Incomplete { required_next },
                _ => ParseStatus::Incomplete {
                    required_next: None,
                },
            },
        }
    }
}

impl Validate for String {
    fn validate<'a>(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        self.as_str().validate(tokens)
    }
}

impl Validate for &str {
    fn validate<'a>(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        let mut iter = tokens.iter();
        let mut chars = self.chars();

        let result = loop {
            let next_char = match iter.peek() {
                Some(c) => c,
                None => {
                    let remaining = chars.as_str().to_string();
                    break if chars.next().is_none() {
                        ParseStatus::Complete(None)
                    } else {
                        ParseStatus::Incomplete {
                            required_next: (!remaining.is_empty()).then_some(remaining),
                        }
                    };
                }
            };

            let my_char = match chars.next() {
                Some(c) => c,
                None => break ParseStatus::Complete(Some(iter.into())),
            };

            if my_char != *next_char {
                break ParseStatus::Invalid;
            }

            let _ = iter.next();
        };

        result
    }
}

/// A parser that will parse an item surrounded by a start and end.
pub struct Between<S: Validate, I: Validate, E: Validate> {
    start: S,
    inner: I,
    end: E,
}

impl<S: Validate, I: Validate, E: Validate> Between<S, I, E> {
    /// Create a new `Between` parser. This parser will parse the start, then the inner, then the end.
    pub fn new(start: S, inner: I, end: E) -> Self {
        Between { start, inner, end }
    }
}

impl<S: Validate, I: Validate, E: Validate> Validate for Between<S, I, E> {
    #[tracing::instrument(skip(self), level = "info")]
    fn validate<'a>(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        (&self.start)
            .then(&self.inner)
            .then(&self.end)
            .validate(tokens)
    }
}

struct Separated<S: Validate, I: Validate> {
    inner: I,
    separator: S,
    min: usize,
    max: usize,
}

impl<S: Validate, I: Validate> Separated<S, I> {
    fn new(inner: I, separator: S, min: usize, max: usize) -> Self {
        Separated {
            inner,
            separator,
            min,
            max,
        }
    }
}

impl<S: Validate, I: Validate> Validate for Separated<S, I> {
    fn validate<'a>(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        let mut tokens = tokens;
        let mut count = 0;
        loop {
            if count >= self.max {
                return ParseStatus::Complete(Some(tokens));
            }
            // first parse an item
            match self.inner.validate(tokens.clone()) {
                // if we get a complete item, then we can parse a separator
                ParseStatus::Complete(Some(new_tokens)) => {
                    count += 1;
                    match self.separator.validate(new_tokens.clone()) {
                        // if we get a complete separator, then we can parse another item
                        ParseStatus::Complete(Some(new_tokens)) => tokens = new_tokens,
                        // if we get a complete separator with no tokens, then we are done
                        ParseStatus::Complete(None) => {
                            if count >= self.min {
                                return ParseStatus::Complete(Some(tokens));
                            } else {
                                return ParseStatus::Incomplete {
                                    required_next: None,
                                };
                            }
                        }
                        // if we get an invalid separator, then this is the end of the list
                        ParseStatus::Invalid => {
                            if count >= self.min {
                                return ParseStatus::Complete(Some(new_tokens));
                            } else {
                                return ParseStatus::Invalid;
                            }
                        }
                        // if we get an incomplete separator, then we need to wait for more tokens
                        ParseStatus::Incomplete { required_next } => {
                            return ParseStatus::Incomplete { required_next }
                        }
                    }
                }
                ParseStatus::Complete(None) => {
                    count += 1;
                    if count >= self.min {
                        // if we get a complete item with no tokens and enough items, then we are done
                        return ParseStatus::Complete(None);
                    } else {
                        return ParseStatus::Invalid;
                    }
                }
                ParseStatus::Invalid => {
                    return ParseStatus::Invalid;
                }
                ParseStatus::Incomplete { required_next } => {
                    return ParseStatus::Incomplete { required_next }
                }
            }
        }
    }
}

#[test]
fn test_string() {
    {
        let tokens = "abcdefghiw";
        let stream = ParseStream::new(tokens);

        let string = "abc";
        assert!(string.validate(stream).is_complete());
    }

    {
        let stream = ParseStream::new("def");

        let string = "def";
        assert!(string.validate(stream).is_complete());
    }

    {
        let stream = ParseStream::new("def");

        let string = "definition";
        assert!(string.validate(stream).is_incomplete());
    }

    {
        let stream = ParseStream::new("dfe");

        let string = "defin";
        assert!(string.validate(stream).is_invalid());
    }
}

#[test]
fn test_separated() {
    let should_parse = [(3, "a,a,a"), (2, "a,a,"), (1, "a,")];
    for (count, tokens) in should_parse {
        let stream = ParseStream::new(tokens);

        let separated = Separated {
            inner: "a",
            separator: ",",
            min: count,
            max: count,
        };

        assert!(dbg!(separated.validate(dbg!(stream))).is_complete());
    }
}

#[test]
fn test_separated_string() {
    let should_parse = [(3, "\"a\",\"a\",\"a\""), (2, "\"a\",\"a\","), (1, "\"a\",")];
    for (count, tokens) in should_parse {
        let stream = ParseStream::new(tokens);

        let separated = Separated {
            inner: ValidateString(1, 3),
            separator: ",",
            min: count,
            max: count,
        };

        assert!(dbg!(separated.validate(dbg!(stream))).is_complete());
    }

    let should_be_incomplete = [(3, "\"a\",\"a\",\"a"), (2, "\"a\",\"a"), (1, "\"_")];
    for (count, tokens) in should_be_incomplete {
        let stream = ParseStream::new(tokens);

        let separated = Separated {
            inner: ValidateString(1, 3),
            separator: ",",
            min: count,
            max: count,
        };

        assert!(dbg!(separated.validate(dbg!(stream))).is_incomplete());
    }
}
