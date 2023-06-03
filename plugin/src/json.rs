use std::{cell::Cell, collections::HashMap, ops::Deref};

#[derive(Debug)]
pub enum Structure {
    Sequence(Box<Structure>),
    Map(HashMap<String, Structure>),
    Num,
    String,
    Either(Box<Structure>, Box<Structure>),
}

impl<'a> Validate<'a> for Structure{
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        match self {
            Structure::Sequence(inner) => {
                let parse_sequence =Seperated::new(
                     inner.as_ref(),
                     ",",
                     None
                );

                let parse_array = Between::new(
                    "[",
                    parse_sequence,
                    "]",
                );

                parse_array.validate(tokens)
            }
            Structure::Map(_map) => {
                let parse_key = SkipSpaces.then(ValidateString);
                let parse_key_colon = parse_key.then(":");
                let parse_value = SkipSpaces.then(ValidateString);
                let parse_kv = parse_key_colon.then(parse_value);

                let parse_map = Seperated::new(
                    parse_kv,
                    ",",
                    None
                );

                let parse_object = Between::new(
                    "{",
                    parse_map,
                    "}",
                );

                parse_object.validate(tokens)
            }
            Structure::Num => {
                let parse_int = ValidateInt;
                SkipSpaces.then(
                    parse_int
                ).validate(tokens)
            },
            Structure::String => {
                let parse_string = ValidateString;
                SkipSpaces.then(
                    parse_string
                ).validate(tokens)
            },
            Structure::Either(left, right) => {
                left.deref().or(right.deref()).validate(tokens)
            }
        }
    }
}

#[test]
fn parse_structured(){
    let parse_array_of_ints = Structure::Sequence(Box::new(Structure::Num));
    
    let tokens = ParseStream::new(&["[1,2,3]"]);
    assert!(parse_array_of_ints.validate(tokens).is_complete());

    let tokens = ParseStream::new(&["[1,2,3"]);
    assert!(parse_array_of_ints.validate(tokens).is_incomplete());

    let tokens = ParseStream::new(&["1,2,3"]);
    assert!(parse_array_of_ints.validate(tokens).is_invalid());

    let parse_array_of_strings = Structure::Sequence(Box::new(Structure::String));
    
    let tokens = ParseStream::new(&[r#"["hello", "world"]"#]);
    assert!(parse_array_of_strings.validate(tokens).is_complete());

    let tokens = ParseStream::new(&[r#"["hello", "world"#]);
    assert!(parse_array_of_strings.validate(tokens).is_incomplete());

    let tokens = ParseStream::new(&["hello, world"]);
    assert!(parse_array_of_strings.validate(tokens).is_invalid());

    let parse_array_of_either = Structure::Sequence(Box::new(Structure::Either(
        Box::new(Structure::Num),
        Box::new(Structure::String),
    )));

    let tokens = ParseStream::new(&[r#"[1,"2",3]"#]);
    assert!(parse_array_of_either.validate(tokens).is_complete());

    let tokens = ParseStream::new(&[r#"[1,"2",3"#]);
    assert!(parse_array_of_either.validate(tokens).is_incomplete());

    let tokens = ParseStream::new(&[r#"1,"2",3"#]);
    assert!(parse_array_of_either.validate(tokens).is_invalid());

    let parse_object = Structure::Map(HashMap::new());

    let tokens = ParseStream::new(&[r#"{"hello": "world"}"#]);
    assert!(parse_object.validate(tokens).is_complete());

    let tokens = ParseStream::new(&[r#"{"hello": "world""#]);
    assert!(parse_object.validate(tokens).is_incomplete());

    let tokens = ParseStream::new(&[r#""hello": "world""#]);
    assert!(parse_object.validate(tokens).is_invalid());
}


struct SkipSpaces;

impl<'a> Validate<'a> for SkipSpaces {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        let mut iter = tokens.iter();

        while let Some(c) = iter.peek() {
            match c {
                ' ' | '\t' | '\n' => {}
                _ => return ParseStatus::Complete(iter.current()),
            }
            let _ = iter.next();
        }

        ParseStatus::Complete(iter.current())
    }
}

struct ValidateInt;

impl<'a> Validate<'a> for ValidateInt {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        let mut iter = tokens.iter();
        
        if iter.peek() == Some('-') {
            let _ = iter.next();
        }
        
        let mut has_decimal = false;
        let mut has_digits = false;

        while let Some(c) = iter.peek() {
            match c {
                '0'..='9'  => {has_digits = true;}
                '.' => {
                    if has_decimal {
                        return ParseStatus::Complete(iter.current());
                    }
                    has_decimal = true;
                }
                _ => {
                    if has_digits {
                        return ParseStatus::Complete(iter.current());
                    } else {
                        return ParseStatus::Invalid;
                    }
                },
            }
            let _ = iter.next();
        }

        ParseStatus::Complete(iter.current())
    }
}

#[test]
fn test_parse_num(){
    let tokens = ParseStream::new(&["-1234"]);
    assert!(ValidateInt.validate(tokens).is_complete());

    let tokens = ParseStream::new(&["1234hello"]);
    assert!(ValidateInt.validate(tokens).is_complete());

    let tokens = ParseStream::new(&["1234.0"]);
    assert!(ValidateInt.validate(tokens).is_complete());

    let tokens = ParseStream::new(&["1234.0.0"]);
    assert!(ValidateInt.validate(tokens).is_complete());
}

struct ValidateString;

impl<'a> Validate<'a> for ValidateString {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        println!("validating string {:?}", tokens);
        let mut iter = tokens.iter();
        let mut escape = false;

        if iter.peek() != Some('"') {
            return ParseStatus::Invalid;
        }
        let _ = iter.next();

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
                    if !escape{
                        let _ = iter.next();
                        return dbg!(ParseStatus::Complete(iter.current()));
                    }
                }
                _ => {}
            }
            let _ = iter.next();
        }

        ParseStatus::Incomplete
    }
}

#[test]
fn test_validate_string() {
    let tokens = ParseStream::new(&["\"hello", "\""]);
    assert!(ValidateString.validate(tokens).is_complete());

    let tokens = ParseStream::new(&["\"hello", "world"]);
    assert!(ValidateString.validate(tokens).is_incomplete());

    let tokens = ParseStream::new(&["hello", "\""]);
    assert!(ValidateString.validate(tokens).is_invalid());
}

#[derive(Debug, PartialEq)]
pub enum ParseStatus<'a> {
    Incomplete,
    Complete(Option<ParseStream<'a>>),
    Invalid,
}

impl ParseStatus<'_> {
    fn is_complete(&self) -> bool {
        matches!(self, ParseStatus::Complete(_))
    }

    pub fn is_invalid(&self) -> bool {
        matches!(self, ParseStatus::Invalid)
    }

    fn is_incomplete(&self) -> bool {
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
        let mut char_index = self.char_index+1;
        loop {
            if let Some(c) = self.tokens.get(token_index) {
                if c.chars().nth(char_index).is_some() {
                    return Some(ParseStream {
                        tokens: self.tokens,
                        token_index,
                        char_index,
                    });
                }
                else {
                    token_index += 1;
                    char_index = 0;
                }
            }
            else {
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

struct Anonymous<F>(pub F);

impl<'a, F> Validate<'a> for Anonymous<F>
where
    F: Fn(ParseStream<'a>) -> ParseStatus<'a>,
{
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        self.0(tokens)
    }
}

impl<'a, V: Validate<'a>> Validate<'a> for &V {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        (*self).validate(tokens)
    }
}

struct Then<'a, A: Validate<'a>, B: Validate<'a>>(A, B, std::marker::PhantomData<&'a ()>);

impl<'a, A: Validate<'a>, B: Validate<'a>> Validate<'a> for Then<'a, A, B> {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        match dbg!(self.0.validate(tokens)) {
            ParseStatus::Complete(Some(tokens)) => {
                dbg!(self.1.validate(tokens))
            }
            ParseStatus::Complete(None) => ParseStatus::Incomplete,
            ParseStatus::Invalid => ParseStatus::Invalid,
            ParseStatus::Incomplete => ParseStatus::Incomplete,
        }
    }
}

struct Or<'a, A: Validate<'a>, B: Validate<'a>>(A, B, std::marker::PhantomData<&'a ()>);

impl<'a, A: Validate<'a>, B: Validate<'a>> Validate<'a> for Or<'a, A, B> {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        match dbg!(self.0.validate(tokens)) {
            ParseStatus::Complete(tokens) => ParseStatus::Complete(tokens),
            ParseStatus::Invalid => dbg!(self.1.validate(tokens)),
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

            let _ =iter.next();
        };

        println!("{}: {:?}", self, result);
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
