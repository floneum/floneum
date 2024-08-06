#[cfg(test)]
use pretty_assertions::assert_eq;

use std::{fmt::Display, fmt::Write};

struct IndentationWriter<'a> {
    indentation: usize,
    writer: &'a mut dyn std::fmt::Write,
}

impl<'a> IndentationWriter<'a> {
    fn new(indentation: usize, writer: &'a mut dyn std::fmt::Write) -> Self {
        Self {
            indentation,
            writer,
        }
    }

    fn with_indent<O>(&mut self, f: impl FnOnce(&mut Self) -> O) -> O {
        self.indentation += 1;
        let out = f(self);
        self.indentation -= 1;
        out
    }
}

impl<'a> std::fmt::Write for IndentationWriter<'a> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        for char in s.chars() {
            self.writer.write_char(char)?;
            if char == '\n' {
                for _ in 0..self.indentation {
                    self.writer.write_char('\t')?;
                }
            }
        }
        Ok(())
    }
}

/// A literal value in a schema
#[derive(Debug, Clone)]
pub enum SchemaLiteral {
    /// A string
    String(String),
    /// A number
    Number(f64),
    /// A boolean
    Boolean(bool),
    /// The null value
    Null,
}

impl Display for SchemaLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchemaLiteral::String(string) => write!(f, "\"{}\"", string),
            SchemaLiteral::Number(number) => write!(f, "{}", number),
            SchemaLiteral::Boolean(boolean) => write!(f, "{}", boolean),
            SchemaLiteral::Null => write!(f, "null"),
        }
    }
}

/// The type of a schema
#[derive(Debug, Clone)]
pub enum SchemaType {
    /// A string schema
    String(StringSchema),
    /// A floating point or integer schema
    Number(NumberSchema),
    /// An integer schema
    Integer(IntegerSchema),
    /// A boolean schema
    Boolean(BooleanSchema),
    /// An array schema
    Array(ArraySchema),
    /// An object schema
    Object(JsonObjectSchema),
    /// An enum schema
    Enum(EnumSchema),
    /// A schema that matches any of the composite schemas
    AnyOf(AnyOfSchema),
    /// A schema that matches one of the composite schemas
    OneOf(OneOfSchema),
    /// A constant schema
    Const(ConstSchema),
    /// An if-then schema
    IfThen(IfThenSchema),
    /// The null schema
    Null,
}

impl SchemaType {
    fn display_with_description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        description: Option<&str>,
    ) -> std::fmt::Result {
        match self {
            SchemaType::String(schema) => schema.display_with_description(f, description),
            SchemaType::Number(schema) => schema.display_with_description(f, description),
            SchemaType::Integer(schema) => schema.display_with_description(f, description),
            SchemaType::Boolean(schema) => schema.display_with_description(f, description),
            SchemaType::Array(schema) => schema.display_with_description(f, description),
            SchemaType::Object(schema) => schema.display_with_description(f, description),
            SchemaType::Enum(schema) => schema.display_with_description(f, description),
            SchemaType::AnyOf(schema) => schema.display_with_description(f, description),
            SchemaType::OneOf(schema) => schema.display_with_description(f, description),
            SchemaType::Const(schema) => schema.display_with_description(f, description),
            SchemaType::IfThen(schema) => schema.display_with_description(f, description),
            SchemaType::Null => match description {
                Some(description) => f.write_fmt(format_args!(
                    "{{\n\t\"description\": \"{description}\",\n\t\"type\": \"null\"\n}}"
                )),
                None => f.write_str("{ \"type\": \"null\" }"),
            },
        }
    }
}

impl Display for SchemaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_with_description(f, None)
    }
}

/// A schema for an conditional schema
#[derive(Debug, Clone)]
pub struct IfThenSchema {
    if_schema: Box<SchemaType>,
    then_schema: Box<SchemaType>,
}

impl IfThenSchema {
    /// Create a new if-then schema
    pub fn new(if_schema: SchemaType, then_schema: SchemaType) -> Self {
        Self {
            if_schema: Box::new(if_schema),
            then_schema: Box::new(then_schema),
        }
    }

    fn display_with_description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        description: Option<&str>,
    ) -> std::fmt::Result {
        f.write_char('{')?;
        {
            let mut writer = IndentationWriter::new(1, f);
            if let Some(description) = description {
                write!(&mut writer, "\n\"description\": \"{description}\",")?;
            }
            writer.write_str("\n\"if\": ")?;
            write!(&mut writer, "{}", self.if_schema)?;
            writer.write_str(",\n\"then\": ")?;
            write!(&mut writer, "{}", self.then_schema)?;
        }
        f.write_str("\n}")
    }
}

impl Display for IfThenSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_with_description(f, None)
    }
}

/// A schema that matches any of the composite schemas
#[derive(Debug, Clone)]
pub struct AnyOfSchema {
    any_of: Vec<SchemaType>,
}

impl AnyOfSchema {
    /// Create a new any of schema
    pub fn new(any_of: impl IntoIterator<Item = SchemaType>) -> Self {
        Self {
            any_of: any_of.into_iter().collect(),
        }
    }

    fn display_with_description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        description: Option<&str>,
    ) -> std::fmt::Result {
        f.write_char('{')?;
        {
            let mut writer = IndentationWriter::new(1, f);
            if let Some(description) = description {
                write!(&mut writer, "\n\"description\": \"{description}\",")?;
            }
            writer.write_str("\n\"anyOf\": [")?;
            if !self.any_of.is_empty() {
                writer.with_indent(|writer| {
                    for (i, schema) in self.any_of.iter().enumerate() {
                        if i > 0 {
                            writer.write_char(',')?;
                        }
                        write!(writer, "\n{}", schema)?;
                    }
                    Ok(())
                })?;
                writer.write_str("\n")?;
            }
            writer.write_str("]")?;
        }
        f.write_str("\n}")
    }
}

impl Display for AnyOfSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_with_description(f, None)
    }
}

/// A schema that matches one of the composite schemas
#[derive(Debug, Clone)]
pub struct OneOfSchema {
    one_of: Vec<SchemaType>,
}

impl OneOfSchema {
    /// Create a new one of schema
    pub fn new(one_of: impl IntoIterator<Item = SchemaType>) -> Self {
        Self {
            one_of: one_of.into_iter().collect(),
        }
    }

    fn display_with_description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        description: Option<&str>,
    ) -> std::fmt::Result {
        f.write_char('{')?;
        {
            let mut writer = IndentationWriter::new(1, f);
            if let Some(description) = description {
                write!(&mut writer, "\n\"description\": \"{description}\",")?;
            }
            writer.write_str("\n\"oneOf\": [")?;
            if !self.one_of.is_empty() {
                writer.with_indent(|writer| {
                    for (i, schema) in self.one_of.iter().enumerate() {
                        if i > 0 {
                            writer.write_char(',')?;
                        }
                        write!(writer, "\n{}", schema)?;
                    }
                    Ok(())
                })?;
                writer.write_str("\n")?;
            }
            writer.write_str("]")?;
        }
        f.write_str("\n}")
    }
}

impl Display for OneOfSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_with_description(f, None)
    }
}

/// A schema for a constant
#[derive(Debug, Clone)]
pub struct ConstSchema {
    value: SchemaLiteral,
}

impl ConstSchema {
    /// Create a new const schema
    pub fn new(value: impl Into<SchemaLiteral>) -> Self {
        Self {
            value: value.into(),
        }
    }

    fn display_with_description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        description: Option<&str>,
    ) -> std::fmt::Result {
        if let Some(description) = description {
            write!(
                f,
                "{{\n\t\"descripiton\": \"{description}\"\n\t\"const\": {}\n}}",
                self.value
            )
        } else {
            write!(f, "{{ \"const\": {} }}", self.value)
        }
    }
}

impl Display for ConstSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_with_description(f, None)
    }
}

#[test]
fn test_const_schema() {
    let schema = ConstSchema::new(SchemaLiteral::String("hello".to_string()));

    assert_eq!(schema.to_string(), "{ \"const\": \"hello\" }");
}

/// A schema for an enum
#[derive(Debug, Clone)]
pub struct EnumSchema {
    variants: Vec<SchemaLiteral>,
}

impl EnumSchema {
    /// Create a new enum schema
    pub fn new(variants: impl IntoIterator<Item = SchemaLiteral>) -> Self {
        Self {
            variants: variants.into_iter().collect(),
        }
    }

    fn display_with_description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        description: Option<&str>,
    ) -> std::fmt::Result {
        f.write_char('{')?;
        {
            let mut writer = IndentationWriter::new(1, f);
            if let Some(description) = description {
                write!(&mut writer, "\n\"description\": \"{description}\",")?;
            }
            writer.write_str("\n\"enum\": [")?;
            {
                for (i, variant) in self.variants.iter().enumerate() {
                    if i > 0 {
                        writer.write_str(", ")?;
                    }
                    write!(writer, "{}", variant)?;
                }
            }
            writer.write_str("]\n")?;
        }
        f.write_str(" }")
    }
}

impl Display for EnumSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_with_description(f, None)
    }
}

/// A schema for a string
#[derive(Debug, Clone)]
pub struct StringSchema {
    /// The length that is valid for the string
    length: Option<std::ops::RangeInclusive<usize>>,
    /// The regex pattern that the string must match
    pattern: Option<String>,
}

impl Schema for String {
    fn schema() -> SchemaType {
        SchemaType::String(StringSchema::new())
    }
}

impl Default for StringSchema {
    fn default() -> Self {
        Self::new()
    }
}

impl StringSchema {
    /// Create a new string schema
    pub fn new() -> Self {
        Self {
            length: None,
            pattern: None,
        }
    }

    /// Set the length range of the string
    pub fn with_length(
        mut self,
        length: impl Into<Option<std::ops::RangeInclusive<usize>>>,
    ) -> Self {
        self.length = length.into();
        self
    }

    /// Set a regex pattern the string must match
    pub fn with_pattern(mut self, pattern: impl ToString) -> Self {
        self.pattern = Some(pattern.to_string());
        self
    }

    fn display_with_description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        description: Option<&str>,
    ) -> std::fmt::Result {
        f.write_char('{')?;
        {
            let mut writer = IndentationWriter::new(1, f);
            if let Some(description) = description {
                write!(&mut writer, "\n\"description\": \"{description}\",")?;
            }
            writer.write_str("\n\"type\": \"string\"")?;
            if let Some(length) = &self.length {
                if *length.start() > 0 {
                    writer.write_fmt(format_args!(",\n\"minLength\": {}", length.start()))?;
                }
                if *length.end() < usize::MAX {
                    writer.write_fmt(format_args!(",\n\"maxLength\": {}", length.end()))?;
                }
            }
            if let Some(pattern) = &self.pattern {
                writer.write_fmt(format_args!(",\n\"pattern\": \"{}\"", pattern))?;
            }
        }
        f.write_str("\n}")
    }
}

impl Display for StringSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_with_description(f, None)
    }
}

/// A schema for a number (floating point or integer)
#[derive(Debug, Clone)]
pub struct NumberSchema {
    /// The range that the number must be in
    range: Option<std::ops::RangeInclusive<f64>>,
}

macro_rules! impl_schema_for_number {
    ($ty:ty) => {
        impl Schema for $ty {
            fn schema() -> SchemaType {
                SchemaType::Number(NumberSchema::new())
            }
        }
    };
}

impl_schema_for_number!(f64);
impl_schema_for_number!(f32);

impl Default for NumberSchema {
    fn default() -> Self {
        Self::new()
    }
}

impl NumberSchema {
    /// Create a new number schema
    pub fn new() -> Self {
        Self { range: None }
    }

    /// Set the range of the number
    pub fn with_range(mut self, range: impl Into<Option<std::ops::RangeInclusive<f64>>>) -> Self {
        self.range = range.into();
        self
    }

    fn display_with_description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        description: Option<&str>,
    ) -> std::fmt::Result {
        match &self.range {
            Some(range) => {
                f.write_char('{')?;
                {
                    let mut writer = IndentationWriter::new(1, f);
                    if let Some(description) = description {
                        write!(&mut writer, "\n\"description\": \"{description}\",")?;
                    }
                    writer.write_str("\n\"type\": \"number\",")?;
                    writer.write_fmt(format_args!("\n\"minimum\": {},", range.start()))?;
                    writer.write_fmt(format_args!("\n\"maximum\": {}", range.end()))?;
                }
                f.write_str("\n}")
            }
            None => f.write_str("{ \"type\": \"number\" }"),
        }
    }
}

impl Display for NumberSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_with_description(f, None)
    }
}

#[test]
fn test_number_schema() {
    let schema = NumberSchema {
        range: Some(0.0..=100.0),
    };

    assert_eq!(
        schema.to_string(),
        "{\n\t\"type\": \"number\",\n\t\"minimum\": 0,\n\t\"maximum\": 100\n}"
    );

    let schema = NumberSchema { range: None };

    assert_eq!(schema.to_string(), "{ \"type\": \"number\" }");
}

/// A schema for an integer
#[derive(Debug, Clone, Default)]

pub struct IntegerSchema;

impl IntegerSchema {
    /// Create a new integer schema
    pub fn new() -> Self {
        Self
    }
}

macro_rules! impl_schema_for_integer {
    ($ty:ty) => {
        impl Schema for $ty {
            fn schema() -> SchemaType {
                SchemaType::Number(NumberSchema::new())
            }
        }
    };
}

impl_schema_for_integer!(i128);
impl_schema_for_integer!(i64);
impl_schema_for_integer!(i32);
impl_schema_for_integer!(i16);
impl_schema_for_integer!(i8);
impl_schema_for_integer!(isize);

impl_schema_for_integer!(u128);
impl_schema_for_integer!(u64);
impl_schema_for_integer!(u32);
impl_schema_for_integer!(u16);
impl_schema_for_integer!(u8);
impl_schema_for_integer!(usize);

impl IntegerSchema {
    fn display_with_description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        description: Option<&str>,
    ) -> std::fmt::Result {
        if let Some(description) = description {
            write!(
                f,
                "{{\n\t\"description\": \"{description}\",\n\t\"type\": \"integer\"\n}}"
            )
        } else {
            f.write_str("{ \"type\": \"integer\" }")
        }
    }
}

impl Display for IntegerSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_with_description(f, None)
    }
}

#[test]
fn test_integer_schema() {
    let schema = IntegerSchema;

    assert_eq!(schema.to_string(), "{ \"type\": \"integer\" }");
}

/// A schema for a boolean
#[derive(Debug, Clone, Default)]
pub struct BooleanSchema;

impl BooleanSchema {
    fn display_with_description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        description: Option<&str>,
    ) -> std::fmt::Result {
        if let Some(description) = description {
            write!(
                f,
                "{{\n\t\"description\": \"{description}\",\n\t\"type\": \"boolean\"\n}}"
            )
        } else {
            f.write_str("{ \"type\": \"boolean\" }")
        }
    }
}

impl Display for BooleanSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_with_description(f, None)
    }
}

#[test]
fn test_boolean_schema() {
    let schema = BooleanSchema;

    assert_eq!(schema.to_string(), "{ \"type\": \"boolean\" }");
}

/// A schema for an array
#[derive(Debug, Clone)]
pub struct ArraySchema {
    items: Box<SchemaType>,
    length: Option<std::ops::RangeInclusive<usize>>,
}

impl<T: Schema> Schema for Vec<T> {
    fn schema() -> SchemaType {
        SchemaType::Array(ArraySchema::new(T::schema()))
    }
}

impl<const N: usize, T: Schema> Schema for [T; N] {
    fn schema() -> SchemaType {
        SchemaType::Array(ArraySchema::new(T::schema()).with_length(N..=N))
    }
}

impl ArraySchema {
    /// Create a new array schema
    pub fn new(items: SchemaType) -> Self {
        Self {
            items: Box::new(items),
            length: None,
        }
    }

    /// Set the length range of the array
    pub fn with_length(
        mut self,
        length: impl Into<Option<std::ops::RangeInclusive<usize>>>,
    ) -> Self {
        self.length = length.into();
        self
    }

    fn display_with_description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        description: Option<&str>,
    ) -> std::fmt::Result {
        f.write_char('{')?;
        {
            let mut writer = IndentationWriter::new(1, f);
            if let Some(description) = description {
                write!(&mut writer, "\n\"description\": \"{description}\",")?;
            }
            writer.write_str("\n\"type\": \"array\"")?;
            writer.write_str(",\n\"items\": ")?;
            write!(&mut writer, "{}", self.items)?;
            if let Some(length) = &self.length {
                if *length.start() > 0 {
                    writer.write_str(",\n\"minItems\": ")?;
                    write!(&mut writer, "{}", length.start())?;
                }
                if *length.end() < usize::MAX {
                    writer.write_str(",\n\"maxItems\": ")?;
                    write!(&mut writer, "{}", length.end())?;
                }
            }
            writer.write_str(",\n\"unevaluatedItems\": false")?;
        }
        f.write_str("\n}")
    }
}

impl Display for ArraySchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_with_description(f, None)
    }
}

#[test]
fn test_array_schema() {
    let schema = ArraySchema {
        items: Box::new(SchemaType::String(StringSchema {
            length: Some(1..=10),
            pattern: None,
        })),
        length: Some(0..=10),
    };

    assert_eq!(schema.to_string(), "{\n\t\"type\": \"array\",\n\t\"items\": {\n\t\t\"type\": \"string\",\n\t\t\"minLength\": 1,\n\t\t\"maxLength\": 10\n\t},\n\t\"maxItems\": 10,\n\t\"unevaluatedItems\": false\n}");

    let schema = ArraySchema {
        items: Box::new(SchemaType::String(StringSchema {
            length: None,
            pattern: None,
        })),
        length: Some(1..=usize::MAX),
    };

    assert_eq!(schema.to_string(), "{\n\t\"type\": \"array\",\n\t\"items\": {\n\t\t\"type\": \"string\"\n\t},\n\t\"minItems\": 1,\n\t\"unevaluatedItems\": false\n}");
    let schema = ArraySchema {
        items: Box::new(SchemaType::String(StringSchema {
            length: None,
            pattern: None,
        })),
        length: None,
    };

    assert_eq!(schema.to_string(), "{\n\t\"type\": \"array\",\n\t\"items\": {\n\t\t\"type\": \"string\"\n\t},\n\t\"unevaluatedItems\": false\n}");
}

/// A schema for an object
#[derive(Debug, Clone)]
pub struct JsonObjectSchema {
    title: Option<String>,
    description: Option<&'static str>,
    properties: Vec<JsonPropertySchema>,
}

impl JsonObjectSchema {
    /// Create a new object schema
    pub fn new(properties: impl IntoIterator<Item = JsonPropertySchema>) -> Self {
        Self {
            title: None,
            description: None,
            properties: properties.into_iter().collect(),
        }
    }

    /// Set the title of the object
    pub fn with_title(mut self, title: impl ToString) -> Self {
        self.title = Some(title.to_string());
        self
    }

    /// Set the description of the object
    pub fn with_description(mut self, description: impl Into<Option<&'static str>>) -> Self {
        self.description = description.into();
        self
    }

    fn display_with_description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        description: Option<&str>,
    ) -> std::fmt::Result {
        f.write_char('{')?;
        {
            let mut writer = IndentationWriter::new(1, f);
            writer.write_char('\n')?;
            if let Some(description) = description {
                writeln!(&mut writer, "\"description\": \"{description}\",")?;
            }
            if let Some(title) = &self.title {
                writer.write_str("\"title\": \"")?;
                writer.write_str(title)?;
                writer.write_str("\",\n")?;
            }
            if let Some(description) = &self.description {
                writer.write_fmt(format_args!("\"description\": \"{}\",\n", description))?;
            }
            writer.write_str("\"type\": \"object\",\n")?;
            writer.write_str("\"properties\": {")?;
            if !self.properties.is_empty() {
                writer.with_indent(|writer| {
                    for (i, property) in self.properties.iter().enumerate() {
                        if i > 0 {
                            writer.write_char(',')?;
                        }
                        write!(writer, "\n{}", property)?;
                    }
                    Ok(())
                })?;
                writer.write_str("\n")?;
            }
            writer.write_str("}")?;
            let required = self
                .properties
                .iter()
                .filter_map(|property| (property.required).then_some(property.name.clone()))
                .collect::<Vec<_>>();
            if !required.is_empty() {
                writer.write_str(",\n\"required\": [")?;
                {
                    for (i, required) in required.iter().enumerate() {
                        if i > 0 {
                            writer.write_str(", ")?;
                        }
                        write!(writer, "\"{}\"", required)?;
                    }
                }
                writer.write_str("]")?;
            }
            writer.write_str(",\n\"additionalProperties\": false")?;
        }
        f.write_str("\n}")
    }
}

impl Display for JsonObjectSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_with_description(f, None)
    }
}

#[test]
fn test_object_schema() {
    let schema = JsonObjectSchema {
        title: Some("Person".to_string()),
        description: Some("A person"),
        properties: vec![
            JsonPropertySchema {
                name: "name".to_string(),
                description: None,
                required: true,
                ty: SchemaType::String(StringSchema {
                    length: Some(1..=10),
                    pattern: None,
                }),
            },
            JsonPropertySchema {
                name: "age".to_string(),
                description: None,
                required: true,
                ty: SchemaType::Number(NumberSchema {
                    range: Some(0.0..=100.0),
                }),
            },
            JsonPropertySchema {
                name: "height".to_string(),
                description: None,
                required: false,
                ty: SchemaType::Number(NumberSchema {
                    range: Some(0.0..=500.0),
                }),
            },
        ],
    };

    assert_eq!(schema.to_string(), "{\n\t\"title\": \"Person\",\n\t\"type\": \"object\",\n\t\"description\": \"A person\",\n\t\"properties\": {\n\t\t\"name\": {\n\t\t\t\"type\": \"string\",\n\t\t\t\"minLength\": 1,\n\t\t\t\"maxLength\": 10\n\t\t},\n\t\t\"age\": {\n\t\t\t\"type\": \"number\",\n\t\t\t\"minimum\": 0,\n\t\t\t\"maximum\": 100\n\t\t},\n\t\t\"height\": {\n\t\t\t\"type\": \"number\",\n\t\t\t\"minimum\": 0,\n\t\t\t\"maximum\": 500\n\t\t}\n\t},\n\t\"required\": [\"name\", \"age\"]\n\t\"additionalProperties\": false\n}");
}

/// A schema for a property of an object
#[derive(Debug, Clone)]
pub struct JsonPropertySchema {
    name: String,
    description: Option<&'static str>,
    required: bool,
    ty: SchemaType,
}

impl JsonPropertySchema {
    /// Create a new property schema
    pub fn new(name: impl ToString, ty: SchemaType) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            required: false,
            ty,
        }
    }

    /// Set the description of the property
    pub fn with_description(mut self, description: impl Into<Option<&'static str>>) -> Self {
        self.description = description.into();
        self
    }

    /// Set whether the property is required
    pub fn with_required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }
}

impl Display for JsonPropertySchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("\"{}\": ", self.name))?;
        self.ty.display_with_description(f, self.description)
    }
}

/// A description of the format of a type
pub trait Schema {
    /// Get the schema for the type
    fn schema() -> SchemaType;
}

impl<T: Schema> Schema for Option<T> {
    fn schema() -> SchemaType {
        SchemaType::OneOf(OneOfSchema::new([SchemaType::Null, T::schema()]))
    }
}

impl<T: Schema> Schema for Box<T> {
    fn schema() -> SchemaType {
        T::schema()
    }
}
