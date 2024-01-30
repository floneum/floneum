use regex_syntax::hir::ClassBytesRange;
use regex_syntax::hir::ClassUnicodeRange;
use std::{any::Any, sync::Arc};

use regex_syntax::{hir::Hir, parse};

use crate::{ArcParser, CreateParserState, LiteralParser, Parser, ParserExt, RepeatParser};

/// A parser that uses a regex pattern to parse input.
pub struct RegexParser {
    parser: ArcParser,
}

impl RegexParser {
    /// Create a new `RegexParser` from a regex pattern.
    pub fn new(regex: &str) -> anyhow::Result<Self> {
        let hir = parse(regex)?;

        Ok(Self {
            parser: create_parser_hir(&hir)?,
        })
    }
}

impl CreateParserState for RegexParser {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        self.parser.create_parser_state()
    }
}

impl Parser for RegexParser {
    type Error = Arc<dyn std::error::Error + Send + Sync>;
    type Output = Arc<dyn Any + Send + Sync>;
    type PartialState = Arc<dyn Any + Send + Sync>;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<crate::ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        self.parser.parse(state, input)
    }
}

fn create_parser_hir(hir: &Hir) -> anyhow::Result<ArcParser> {
    Ok(match hir.kind() {
        regex_syntax::hir::HirKind::Empty => todo!(),
        #[allow(clippy::unnecessary_to_owned)]
        regex_syntax::hir::HirKind::Literal(literal) => {
            LiteralParser::new(String::from_utf8_lossy(&literal.0).to_string()).boxed()
        }
        regex_syntax::hir::HirKind::Class(class) => match class.literal() {
            Some(literal) =>
            {
                #[allow(clippy::unnecessary_to_owned)]
                LiteralParser::new(String::from_utf8_lossy(&literal).to_string()).boxed()
            }
            None => match class {
                regex_syntax::hir::Class::Unicode(ranges) => {
                    let ranges: Vec<_> = ranges.iter().cloned().collect();

                    UnicodeRangesParser { range: ranges }.boxed()
                }
                regex_syntax::hir::Class::Bytes(ranges) => {
                    let ranges: Vec<_> = ranges.iter().cloned().collect();

                    BytesRangesParser { range: ranges }.boxed()
                }
            },
        },
        regex_syntax::hir::HirKind::Look(_) => anyhow::bail!("Look is not supported"),
        regex_syntax::hir::HirKind::Repetition(repetition) => {
            if !repetition.greedy {
                return Err(anyhow::anyhow!("Non-greedy repetition is not supported"));
            }

            let parser = create_parser_hir(&repetition.sub)?;
            let min = repetition.min as usize;
            let max = match repetition.max {
                Some(max) => max as usize,
                None => usize::MAX,
            };

            RepeatParser::new(parser, min..=max).boxed()
        }
        regex_syntax::hir::HirKind::Capture(_) => anyhow::bail!("Capture is not supported"),
        regex_syntax::hir::HirKind::Concat(concat) => {
            let mut parsers = concat.iter().map(create_parser_hir);

            let mut parser = match parsers.next() {
                Some(first) => first?,
                None => return Ok(LiteralParser::new("").boxed()),
            };

            for next in parsers {
                parser = parser.then(next?).boxed();
            }

            parser
        }
        regex_syntax::hir::HirKind::Alternation(or) => {
            let mut parsers = or.iter().map(create_parser_hir);

            let mut parser = match parsers.next() {
                Some(first) => first?,
                None => return Ok(LiteralParser::new("").boxed()),
            };

            for next in parsers {
                parser = parser.or(next?).boxed();
            }

            parser
        }
    })
}

#[derive(Debug)]
struct UnmatchedCharRange;

impl std::fmt::Display for UnmatchedCharRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unmatched character range")
    }
}

impl std::error::Error for UnmatchedCharRange {}

struct UnicodeRangesParser {
    range: Vec<ClassUnicodeRange>,
}

impl CreateParserState for UnicodeRangesParser {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {}
}

impl Parser for UnicodeRangesParser {
    type Error = UnmatchedCharRange;
    type Output = ();
    type PartialState = ();

    fn parse<'a>(
        &self,
        _state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<crate::ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        let mut iter = std::str::from_utf8(input).unwrap().char_indices();

        let (i, c) = match iter.next() {
            Some((i, c)) => (i, c),
            None => {
                return Ok(crate::ParseResult::Incomplete {
                    new_state: (),
                    required_next: "".into(),
                });
            }
        };

        let mut found = false;
        for range in &self.range {
            if range.start() <= c && c <= range.end() {
                found = true;
                break;
            }
        }

        if found {
            Ok(crate::ParseResult::Finished {
                result: (),
                remaining: &input[i..],
            })
        } else {
            Err(UnmatchedCharRange)
        }
    }
}

struct BytesRangesParser {
    range: Vec<ClassBytesRange>,
}

impl CreateParserState for BytesRangesParser {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {}
}

impl Parser for BytesRangesParser {
    type Error = UnmatchedCharRange;
    type Output = ();
    type PartialState = ();

    fn parse<'a>(
        &self,
        _state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<crate::ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        let mut iter = std::str::from_utf8(input).unwrap().char_indices();

        let (i, c) = match iter.next() {
            Some((i, c)) => (i, c),
            None => {
                return Ok(crate::ParseResult::Incomplete {
                    new_state: (),
                    required_next: "".into(),
                });
            }
        };
        // try to convert char to u8
        let c: u8 = match c.try_into() {
            Ok(c) => c,
            Err(_) => {
                return Ok(crate::ParseResult::Finished {
                    result: (),
                    remaining: &input[i..],
                })
            }
        };
        let mut found = false;
        for range in &self.range {
            if range.start() <= c && c <= range.end() {
                found = true;
                break;
            }
        }

        if found {
            Ok(crate::ParseResult::Finished {
                result: (),
                remaining: &input[i..],
            })
        } else {
            Err(UnmatchedCharRange)
        }
    }
}

#[test]
fn test_regex_parser() {
    let parser = RegexParser::new(r"abc").unwrap();

    let result = parser.parse(&parser.create_parser_state(), b"abc").unwrap();
    println!(
        "{:?}",
        matches!(result, crate::ParseResult::Finished { .. })
    );

    let result = parser.parse(&parser.create_parser_state(), b"ab").unwrap();
    println!(
        "{:?}",
        matches!(result, crate::ParseResult::Incomplete { .. })
    );

    let result = parser
        .parse(&parser.create_parser_state(), b"abcd")
        .unwrap();
    println!(
        "{:?}",
        matches!(result, crate::ParseResult::Finished { .. })
    );

    let parser = RegexParser::new(r"[a-z]").unwrap();

    let result = parser.parse(&parser.create_parser_state(), b"a").unwrap();
    println!(
        "{:?}",
        matches!(result, crate::ParseResult::Finished { .. })
    );

    let result = parser.parse(&parser.create_parser_state(), b"z").unwrap();
    println!(
        "{:?}",
        matches!(result, crate::ParseResult::Finished { .. })
    );

	let parser = RegexParser::new(r"[a-z]*").unwrap();

	let result = parser.parse(&parser.create_parser_state(), b"abc").unwrap();
	println!(
		"{:?}",
		matches!(result, crate::ParseResult::Finished { .. })
	);

	let result = parser.parse(&parser.create_parser_state(), b"123");
	println!(
		"{:?}",
		result.is_err()
	);

	let result = parser.parse(&parser.create_parser_state(), b"abc123").unwrap();
	println!(
		"{:?}",
		matches!(result, crate::ParseResult::Finished { .. })
	);
}
