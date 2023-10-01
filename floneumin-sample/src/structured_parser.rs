use std::iter::Peekable;use std::str::Chars;
use std::{
    cell::{ RefCell},
    ops::Deref,
};

#[derive(Debug, Clone)]
pub enum StructureParser {
    Literal(String),
    Sequence {
        item: Box<StructureParser>,
        separator: Box<StructureParser>,
        min_len: u64,
        max_len: u64,
    },
    Num {
        min: f64,
        max: f64,
        integer: bool,
    },
    String {
        min_len: u64,
        max_len: u64,
    },
    Either {
        first: Box<StructureParser>,
        second: Box<StructureParser>,
    },
    Then {
        first: Box<StructureParser>,
        second: Box<StructureParser>,
    },
}

impl<'a> Validate<'a> for StructureParser {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        match self {
            StructureParser::Literal(text) => text.validate(tokens),
            StructureParser::Sequence {
                item,
                separator,
                min_len,
                max_len,
            } => {
                let parse_sequence = Seperated::new(
                    item.as_ref(),
                    separator.as_ref(),
                    *min_len as usize,
                    *max_len as usize,
                );

                let parse_array = Between::new("[", parse_sequence, "]");

                parse_array.validate(tokens)
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

impl<'a> Validate<'a> for ValidateInt {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
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
                            return ParseStatus::Complete(Some(iter.into()))
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
                        return ParseStatus::Complete(Some(iter.into()))
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
    let tokens = ParseStream::new(&["-1234 "]);
    assert!(ValidateInt {
        min: -2000.,
        max: 2000.,
        integer: true,
    }
    .validate(tokens)
    .is_complete());

    let tokens = ParseStream::new(&["1234hello"]);
    assert!(ValidateInt {
        min: -2000.,
        max: 2000.,
        integer: true,
    }
    .validate(tokens)
    .is_complete());

    let tokens = ParseStream::new(&["1234.0 "]);
    assert!(ValidateInt {
        min: -2000.,
        max: 2000.,
        integer: false,
    }
    .validate(tokens)
    .is_complete());

    let tokens = ParseStream::new(&["1234.0.0"]);
    assert!(ValidateInt {
        min: -2000.,
        max: 2000.,
        integer: false,
    }
    .validate(tokens)
    .is_complete());
}

struct ValidateString(u64, u64);

impl<'a> Validate<'a> for ValidateString {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        let min_len = self.0;
        let max_len = self.1;
        let mut iter = tokens.iter();
        let mut escape = false;

        if iter.peek().copied() != Some('"') {
            return ParseStatus::Invalid;
        }
        let _ = iter.next();
        let mut string_length = 0;

        while let Some(c) = iter.peek() {
            if escape {
                escape = false;
                continue;
            }

            match c {
                '\\' => {
                    escape = true;
                }
                '"' => {
                    if !escape {
                        if string_length < min_len {
                            return ParseStatus::Invalid;
                        }
                        let _ = iter.next();
                        return ParseStatus::Complete(Some(iter.into()));
                    }
                    string_length += 1;
                }
                _ => {
                    string_length += 1;
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
    let tokens = ParseStream::new(&["\"hello", "\""]);
    assert!(ValidateString(5, 5).validate(tokens).is_complete());

    let tokens = ParseStream::new(&["\"hello", "world"]);
    assert!(ValidateString(5, 50).validate(tokens).is_incomplete());

    let tokens = ParseStream::new(&["hello", "\""]);
    assert!(ValidateString(5, 5).validate(tokens).is_invalid());
}

#[derive(Debug)]
pub enum ParseStatus<'a> {
    Incomplete { required_next: Option<String> },
    Complete(Option<ParseStream<'a>>),
    Invalid,
}

#[allow(unused)]
impl ParseStatus<'_> {
    pub fn is_complete(&self) -> bool {
        matches!(self, ParseStatus::Complete(_))
    }

    pub fn is_invalid(&self) -> bool {
        matches!(self, ParseStatus::Invalid)
    }

    pub fn is_incomplete(&self) -> bool {
        matches!(self, ParseStatus::Incomplete { .. })
    }
}

#[derive(Debug, Clone)]
pub struct ParseStream<'a> {
    chars:Peekable< Chars<'a>>,
}

impl<'a> From<Peekable<Chars<'a>>> for ParseStream<'a> {
    fn from(chars: Peekable<Chars<'a>>) -> Self {
        Self { chars }
    }
}

impl<'a> ParseStream<'a> {
    pub fn new(token: &'a str) -> Self {
        Self {
            chars: token.chars().peekable(),
        }
    }

    fn iter(&self) -> Peekable<Chars<'a>> {
        self.chars.clone()
    }
}

pub trait Validate<'a> {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a>;

    fn or<V: Validate<'a>>(self, other: V) -> Or<'a, Self, V>
    where
        Self: Sized,
    {
        Or(self, other, std::marker::PhantomData)
    }

    fn then<V: Validate<'a>>(self, other: V) -> Then<'a, Self, V>
    where
        Self: Sized,
    {
        Then(self, other, std::marker::PhantomData)
    }

    fn boxed(self) -> BoxedValidate<'a>
    where
        Self: Sized + 'a,
    {
        BoxedValidate(Box::new(self))
    }
}

pub struct BoxedValidate<'a>(pub(crate) Box<dyn Validate<'a> + 'a>);

impl<'a> Validate<'a> for BoxedValidate<'a> {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        self.0.validate(tokens)
    }
}

impl<'a, V: Validate<'a>> Validate<'a> for &V {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
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

impl<'a, F> Validate<'a> for Anonymous<F>
where
    F: FnMut(ParseStream<'a>) -> ParseStatus<'a>,
{
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        (self.0.borrow_mut())(tokens)
    }
}

pub struct Then<'a, A: Validate<'a>, B: Validate<'a>>(A, B, std::marker::PhantomData<&'a ()>);

impl<'a, A: Validate<'a>, B: Validate<'a>> Validate<'a> for Then<'a, A, B> {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
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

pub struct Or<'a, A: Validate<'a>, B: Validate<'a>>(A, B, std::marker::PhantomData<&'a ()>);

impl<'a, A: Validate<'a>, B: Validate<'a>> Validate<'a> for Or<'a, A, B> {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
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

impl<'a> Validate<'a> for String {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        self.as_str().validate(tokens)
    }
}

impl<'a> Validate<'a> for &str {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
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

struct Between<'a, S: Validate<'a>, I: Validate<'a>, E: Validate<'a>> {
    start: S,
    inner: I,
    end: E,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, S: Validate<'a>, I: Validate<'a>, E: Validate<'a>> Between<'a, S, I, E> {
    fn new(start: S, inner: I, end: E) -> Self {
        Between {
            start,
            inner,
            end,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, S: Validate<'a>, I: Validate<'a>, E: Validate<'a>> Validate<'a> for Between<'a, S, I, E> {
    #[tracing::instrument(skip(self), level = "info")]
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        (&self.start)
            .then(&self.inner)
            .then(&self.end)
            .validate(tokens)
    }
}

struct Seperated<'a, S: Validate<'a>, I: Validate<'a>> {
    inner: I,
    separator: S,
    min: usize,
    max: usize,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, S: Validate<'a>, I: Validate<'a>> Seperated<'a, S, I> {
    fn new(inner: I, separator: S, min: usize, max: usize) -> Self {
        Seperated {
            inner,
            separator,
            min,
            max,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, S: Validate<'a>, I: Validate<'a>> Validate<'a> for Seperated<'a, S, I> {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
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
                    if count >= self.min {
                        // if we get a complete item with no tokens and enough items, then we are done
                        return ParseStatus::Complete(None);
                    } else {
                        return ParseStatus::Invalid;
                    }
                }
                ParseStatus::Invalid => return ParseStatus::Invalid,
                ParseStatus::Incomplete { required_next } => {
                    return ParseStatus::Incomplete { required_next }
                }
            }
            count += 1;
        }
    }
}

#[test]
fn test_parse_stream() {
    let tokens = &["abc", "def", "ghi"];
    let stream = ParseStream {
        tokens,
        token_index: 0,
        char_index: 0,
    };

    let mut iter = stream.iter();
    assert_eq!(iter.next(), Some('a'));
    assert_eq!(iter.next(), Some('b'));
    assert_eq!(iter.next(), Some('c'));
    assert_eq!(iter.next(), Some('d'));
    assert_eq!(iter.next(), Some('e'));
    assert_eq!(iter.next(), Some('f'));
    assert_eq!(iter.next(), Some('g'));
    assert_eq!(iter.next(), Some('h'));
    assert_eq!(iter.next(), Some('i'));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_string() {
    {
        let tokens = &["abc", "def", "ghiw"];
        let stream = ParseStream {
            tokens,
            token_index: 0,
            char_index: 0,
        };

        let string = "abc";
        assert!(string.validate(stream).is_complete());
    }

    {
        let tokens = &["d", "ef"];
        let stream = ParseStream {
            tokens,
            token_index: 0,
            char_index: 0,
        };

        let string = "def";
        assert!(string.validate(stream).is_complete());
    }

    {
        let tokens = &["d", "ef"];
        let stream = ParseStream {
            tokens,
            token_index: 0,
            char_index: 0,
        };

        let string = "definition";
        assert!(string.validate(stream).is_incomplete());
    }

    {
        let tokens = &["dfe"];
        let stream = ParseStream {
            tokens,
            token_index: 0,
            char_index: 0,
        };

        let string = "defin";
        assert!(string.validate(stream).is_invalid());
    }
}

#[test]
fn test_seperated() {
    let should_parse = [&["a,a,a"], &["a,a,"], &["a,"]];
    for tokens in should_parse {
        let stream = ParseStream {
            tokens,
            token_index: 0,
            char_index: 0,
        };

        let seperated = Seperated {
            inner: "a",
            separator: ",",
            min: 6,
            max: 6,
            _phantom: std::marker::PhantomData,
        };

        assert!(seperated.validate(stream).is_complete());
    }
}
