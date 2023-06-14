use std::{
    cell::{Cell, RefCell},
    collections::{HashMap, HashSet},
    ops::Deref,
    rc::Rc,
};

#[derive(Debug, Clone)]
pub enum Structure {
    Sequence(Box<Structure>),
    Map(StructureMap),
    Num { min: f64, max: f64, integer: bool },
    String(u64, u64),
    Bool,
    Null,
    Either(Box<Structure>, Box<Structure>),
}

#[derive(Debug, Clone)]
pub struct StructureMap(pub HashMap<String, Structure>);

impl<'a> Validate<'a> for Structure {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        match self {
            Structure::Sequence(inner) => {
                let parse_sequence = Seperated::new(inner.as_ref(), ",", None);

                let parse_array = Between::new("[", parse_sequence, "]");

                parse_array.validate(tokens)
            }
            Structure::Bool => {
                let true_validator = "true";
                let false_validator = "false";
                true_validator.or(false_validator).validate(tokens)
            }
            Structure::Map(map) => map.validate(tokens),
            Structure::Num { min, max, integer } => {
                let parse_int = ValidateInt {
                    min: *min,
                    max: *max,
                    integer: *integer,
                };
                parse_int.validate(tokens)
            }
            Structure::String(min_len, max_len) => {
                let parse_string = ValidateString(*min_len, *max_len);
                parse_string.validate(tokens)
            }
            Structure::Null => {
                let null_validator = "null";
                null_validator.validate(tokens)
            }
            Structure::Either(left, right) => left.deref().or(right.deref()).validate(tokens),
        }
    }
}

#[test]
fn parse_structured() {
    let parse_array_of_ints = Structure::Sequence(Box::new(Structure::Num {
        min: 1.,
        max: 3.,
        integer: true,
    }));

    let tokens = ParseStream::new(&["[1,2,3]"]);
    assert!(parse_array_of_ints.validate(tokens).is_complete());

    let tokens = ParseStream::new(&["[1,2,3"]);
    assert!(parse_array_of_ints.validate(tokens).is_incomplete());

    let tokens = ParseStream::new(&["1,2,3"]);
    assert!(parse_array_of_ints.validate(tokens).is_invalid());

    let parse_array_of_strings = Structure::Sequence(Box::new(Structure::String(5, 5)));

    let tokens = ParseStream::new(&[r#"["hello","world"]"#]);
    assert!(parse_array_of_strings.validate(tokens).is_complete());

    let tokens = ParseStream::new(&[r#"["hello","world"#]);
    assert!(parse_array_of_strings.validate(tokens).is_incomplete());

    let tokens = ParseStream::new(&["hello, world"]);
    assert!(parse_array_of_strings.validate(tokens).is_invalid());

    let parse_array_of_either = Structure::Sequence(Box::new(Structure::Either(
        Box::new(Structure::Num {
            min: 1.,
            max: 3.,
            integer: true,
        }),
        Box::new(Structure::String(0, 1)),
    )));

    let tokens = ParseStream::new(&[r#"[1,"2",3]"#]);
    assert!(parse_array_of_either.validate(tokens).is_complete());

    let tokens = ParseStream::new(&[r#"[1,"2",3"#]);
    assert!(parse_array_of_either.validate(tokens).is_incomplete());

    let tokens = ParseStream::new(&[r#"1,"2",3"#]);
    assert!(parse_array_of_either.validate(tokens).is_invalid());

    let parse_object = Structure::Map(StructureMap(
        vec![
            ("hello".to_string(), Structure::String(5, 5)),
            (
                "world".to_string(),
                Structure::Num {
                    min: 0.,
                    max: 1.,
                    integer: true,
                },
            ),
        ]
        .into_iter()
        .collect(),
    ));

    let tokens = ParseStream::new(&[r#"{"hello":"world","world":1}"#]);
    assert!(parse_object.validate(tokens).is_complete());

    let tokens = ParseStream::new(&[r#"{"hello":"world","world":1"#]);
    assert!(parse_object.validate(tokens).is_incomplete());

    let tokens = ParseStream::new(&[r#""hello":"world","world":1}"#]);
    assert!(parse_object.validate(tokens).is_invalid());
}

impl<'a> Validate<'a> for StructureMap {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        let keys_parsed = Rc::new(RefCell::new(HashSet::new()));
        let parse_kv = Anonymous::new({
            let keys_parsed = keys_parsed.clone();
            move |tokens: ParseStream<'a>| {
                let mut keys_parsed = keys_parsed.borrow_mut();
                let (key, status) = parse_string(tokens);

                match status {
                    ParseStatus::Complete(Some(tokens)) => {
                        let Some(value_structure) = self.0.get(&key) else {
                        return ParseStatus::Invalid;
                    };
                        keys_parsed.insert(key);

                        let parse_colon = ":";

                        let parse_key = value_structure;

                        let parse_kv = parse_colon.then(parse_key);
                        parse_kv.validate(tokens)
                    }
                    ParseStatus::Complete(None) => {
                        if self.0.get(&key).is_some() {
                            keys_parsed.insert(key);
                            ParseStatus::Incomplete
                        } else {
                            ParseStatus::Invalid
                        }
                    }
                    ParseStatus::Incomplete => {
                        // check if any of the keys match the first part of the string
                        // if so, return incomplete
                        // otherwise, return invalid
                        if self
                            .0
                            .keys()
                            .any(|k| !keys_parsed.contains(k) && k.starts_with(&key))
                        {
                            ParseStatus::Incomplete
                        } else {
                            ParseStatus::Invalid
                        }
                    }
                    _ => status,
                }
            }
        });

        let parse_closing_brace_if_complete = Anonymous::new(move |tokens: ParseStream<'a>| {
            if keys_parsed.borrow().len() == self.0.len() {
                let parse_closing_brace = "}";
                parse_closing_brace.validate(tokens)
            } else {
                ParseStatus::Invalid
            }
        });

        let parse_map = Seperated::new(parse_kv, ",", None);

        let parse_object = Between::new("{", parse_map, parse_closing_brace_if_complete);

        parse_object.validate(tokens)
    }
}

fn parse_string(tokens: ParseStream) -> (String, ParseStatus) {
    let mut iter = tokens.iter();
    let mut string = String::new();

    if iter.peek() != Some('"') {
        return (string, ParseStatus::Invalid);
    }
    let _ = iter.next();

    while let Some(c) = iter.peek() {
        match c {
            '"' => {
                let _ = iter.next();
                return (string, ParseStatus::Complete(iter.current()));
            }
            _ => {
                string.push(c);
            }
        }
        let _ = iter.next();
    }

    (string, ParseStatus::Incomplete)
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

        let is_negative = iter.peek() == Some('-');
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
                        if real_number > max || real_number < min {
                            return ParseStatus::Invalid;
                        }
                        return ParseStatus::Complete(iter.current());
                    }
                    has_decimal = true;
                    decimal_place = 0.1;
                }
                _ => {
                    if has_digits {
                        if real_number > max || real_number < min {
                            return ParseStatus::Invalid;
                        }
                        return ParseStatus::Complete(iter.current());
                    } else {
                        return ParseStatus::Invalid;
                    }
                }
            }
            let _ = iter.next();
        }

        ParseStatus::Incomplete
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

        if iter.peek() != Some('"') {
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
                        return ParseStatus::Complete(iter.current());
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

        ParseStatus::Incomplete
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

#[derive(Debug, PartialEq)]
pub enum ParseStatus<'a> {
    Incomplete,
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
        matches!(self, ParseStatus::Incomplete)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParseStream<'a> {
    tokens: &'a [&'a str],
    token_index: usize,
    char_index: usize,
}

impl<'a> ParseStream<'a> {
    pub fn new(tokens: &'a [&'a str]) -> Self {
        Self {
            tokens,
            token_index: 0,
            char_index: 0,
        }
    }

    fn iter(&self) -> ParseStreamIter<'a> {
        ParseStreamIter {
            stream: Cell::new(Some(*self)),
        }
    }

    fn peek(self) -> Option<char> {
        let token_index = self.token_index;
        let char_index = self.char_index;
        if let Some(c) = self.tokens.get(token_index) {
            if let Some(c) = c.chars().nth(char_index) {
                return Some(c);
            }
        }
        None
    }

    fn take(self) -> Option<ParseStream<'a>> {
        let mut token_index = self.token_index;
        let mut char_index = self.char_index + 1;
        loop {
            if let Some(c) = self.tokens.get(token_index) {
                if c.chars().nth(char_index).is_some() {
                    return Some(ParseStream {
                        tokens: self.tokens,
                        token_index,
                        char_index,
                    });
                } else {
                    token_index += 1;
                    char_index = 0;
                }
            } else {
                return None;
            }
        }
    }
}

struct ParseStreamIter<'a> {
    stream: Cell<Option<ParseStream<'a>>>,
}

impl<'a> ParseStreamIter<'a> {
    fn peek(&self) -> Option<char> {
        self.stream.get()?.peek()
    }

    fn current(&self) -> Option<ParseStream<'a>> {
        self.stream.get()
    }
}

impl<'a> Iterator for ParseStreamIter<'a> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        let stream = self.stream.get()?;
        let next = stream.peek();
        self.stream.set(stream.take());
        next
    }
}

pub trait Validate<'a> {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a>;

    fn or<V: Validate<'a>>(&self, other: V) -> Or<'a, &Self, V>
    where
        Self: Sized,
    {
        Or(self, other, std::marker::PhantomData)
    }

    fn then<V: Validate<'a>>(&self, other: V) -> Then<'a, &Self, V>
    where
        Self: Sized,
    {
        Then(self, other, std::marker::PhantomData)
    }
}

impl<'a, V: Validate<'a>> Validate<'a> for &V {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        (*self).validate(tokens)
    }
}

struct Anonymous<F>(RefCell<F>);

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
            ParseStatus::Complete(None) => ParseStatus::Incomplete,
            ParseStatus::Invalid => ParseStatus::Invalid,
            ParseStatus::Incomplete => ParseStatus::Incomplete,
        }
    }
}

pub struct Or<'a, A: Validate<'a>, B: Validate<'a>>(A, B, std::marker::PhantomData<&'a ()>);

impl<'a, A: Validate<'a>, B: Validate<'a>> Validate<'a> for Or<'a, A, B> {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        match self.0.validate(tokens) {
            ParseStatus::Complete(tokens) => ParseStatus::Complete(tokens),
            ParseStatus::Invalid => self.1.validate(tokens),
            ParseStatus::Incomplete => ParseStatus::Incomplete,
        }
    }
}

impl<'a> Validate<'a> for &str {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        let mut iter = tokens.iter();
        let mut chars = self.chars();

        let result = loop {
            let my_char = match chars.next() {
                Some(c) => c,
                None => break ParseStatus::Complete(iter.current()),
            };

            let next_char = match iter.peek() {
                Some(c) => c,
                None => break ParseStatus::Incomplete,
            };

            if my_char != next_char {
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
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        self.start
            .then(&self.inner)
            .then(&self.end)
            .validate(tokens)
    }
}

struct Seperated<'a, S: Validate<'a>, I: Validate<'a>> {
    inner: I,
    seperator: S,
    max: Option<usize>,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, S: Validate<'a>, I: Validate<'a>> Seperated<'a, S, I> {
    fn new(inner: I, seperator: S, max: Option<usize>) -> Self {
        Seperated {
            inner,
            seperator,
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
            if let Some(max) = self.max {
                if count >= max {
                    return ParseStatus::Complete(Some(tokens));
                }
            }
            // first parse an item
            match self.inner.validate(tokens) {
                // if we get a complete item, then we can parse a seperator
                ParseStatus::Complete(Some(new_tokens)) => {
                    match self.seperator.validate(new_tokens) {
                        // if we get a complete seperator, then we can parse another item
                        ParseStatus::Complete(Some(new_tokens)) => tokens = new_tokens,
                        // if we get a complete seperator with no tokens, then we are done
                        ParseStatus::Complete(None) => return ParseStatus::Complete(Some(tokens)),
                        // if we get an invalid seperator, then this is the end of the list
                        ParseStatus::Invalid => return ParseStatus::Complete(Some(new_tokens)),
                        // if we get an incomplete seperator, then we need to wait for more tokens
                        ParseStatus::Incomplete => return ParseStatus::Incomplete,
                    }
                }
                // if we get a complete item with no tokens, then we are done
                ParseStatus::Complete(None) => return ParseStatus::Complete(None),
                ParseStatus::Invalid => return ParseStatus::Invalid,
                ParseStatus::Incomplete => return ParseStatus::Incomplete,
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
            seperator: ",",
            max: None,
            _phantom: std::marker::PhantomData,
        };

        assert!(seperated.validate(stream).is_complete());
    }
}
