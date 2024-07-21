use std::collections::HashMap;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};
use syn::{ext::IdentExt, parse_macro_input, DeriveInput, Field, Ident, LitStr};
use syn::{DataEnum, Fields, LitInt};

/// Derive a default JSON parser for a unit value, struct or enum.
///
/// # Examples
///
/// You can derive a parser for a struct with fields that implement the `Parse` trait:
///
/// ```rust
/// # use kalosm::language::*;
/// #[derive(Parse, Debug, Clone, PartialEq)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// let parser = Person::new_parser();
/// let state = parser.create_parser_state();
/// let person = parser.parse(&state, b"{ \"name\": \"John\", \"age\": 30 } ").unwrap().unwrap_finished();
/// assert_eq!(person.name, "John");
/// assert_eq!(person.age, 30);
/// ```
///
/// Or an enum with unit variants:
/// ```rust
/// # use kalosm::language::*;
/// #[derive(Parse, Debug, Clone, PartialEq)]
/// enum Color {
///     Red,
///     Blue,
///     Green,
/// }
///
/// let parser = Color::new_parser();
/// let state = parser.create_parser_state();
/// let color = parser.parse(&state, b"\"Red\" ").unwrap().unwrap_finished();
/// assert_eq!(color, Color::Red);
/// ```
///
/// You can even derive Parse for an enum with data variants:
/// ```rust
/// # use kalosm::language::*;
/// #[derive(Parse, Debug, Clone, PartialEq)]
/// enum Action {
///     Search { query: String },
///     Quit,
/// }
///
/// let parser = Action::new_parser();
/// let state = parser.create_parser_state();
/// let action = parser.parse(&state, b"{ \"type\": \"Search\", \"data\": { \"query\": \"my query\" } } ").unwrap().unwrap_finished();
/// assert_eq!(action, Action::Search { query: "my query".to_string() });
/// ```
///
/// ## Attributes
///
/// The `#[parse]` attribute modifies the default behavior of the parser. It can be used in the following forms:
///
/// - `#[parse(rename = "name")]` renames the field or type to `name` (defaults to the field name)
///
/// ```rust
/// # use kalosm::language::*;
/// #[derive(Parse, Clone)]
/// struct Person {
///     #[parse(rename = "full name")]
///     name: String,
///     age: u32,
/// }
///
/// #[derive(Parse, Clone)]
/// enum Color {
///     #[parse(rename = "red")]
///     Red,
///     Blue,
///     Green,
/// }
/// ```
///
/// - `#[parse(with = expression)]` uses the expression to parse the field (defaults to the parser provided by the `Parse` implementation for the field type)
///
/// ```rust
/// # use kalosm::language::*;
/// #[derive(Parse, Clone)]
/// struct Person {
///     #[parse(with = StringParser::new(1..=10))]
///     name: String,
///     age: u32,
/// }
/// ```
///
/// - `#[parse(tag = "tag")]` changes the name of the tag for enum variants (defaults to "type")
///
/// ```rust
/// # use kalosm::language::*;
/// #[derive(Parse, Clone)]
/// #[parse(tag = "action")]
/// enum Action {
///     Search { query: String },
///     Quit,
/// }
/// ```
///
/// - `#[parse(content = "content")]` changes the name of the content for enum variants (defaults to "data")
///
/// ```rust
/// # use kalosm::language::*;
/// #[derive(Parse, Clone)]
/// #[parse(content = "arguments")]
/// enum Action {
///     Search { query: String },
///     Quit,
/// }
/// ```
#[proc_macro_derive(Parse, attributes(parse))]
pub fn derive_parse(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    match input.data {
        syn::Data::Struct(data) => match data.fields {
            syn::Fields::Named(fields) => {
                let ty = input.ident;
                if fields.named.is_empty() {
                    return TokenStream::from(impl_unit_parser(
                        &input.attrs,
                        &ty,
                        quote! { Self {} },
                    ));
                }

                let field_names = fields.named.iter().map(|f| f.ident.as_ref().unwrap());
                let construct = quote! {
                    Self {
                        #(
                            #field_names
                        ),*
                    }
                };

                let parser = match field_parser(&fields.named.iter().collect::<Vec<_>>(), construct)
                {
                    Ok(parser) => parser,
                    Err(err) => return err.to_compile_error().into(),
                };
                let expanded = quote! {
                    impl kalosm_sample::Parse for #ty {
                        fn new_parser() -> impl kalosm_sample::SendCreateParserState<Output = Self> {
                            #parser
                        }
                    }
                };

                TokenStream::from(expanded)
            }
            syn::Fields::Unit => {
                let ty = input.ident;
                TokenStream::from(impl_unit_parser(&input.attrs, &ty, quote! { Self }))
            }
            _ => syn::Error::new(
                input.ident.span(),
                "Only structs with named fields are supported",
            )
            .to_compile_error()
            .into(),
        },
        syn::Data::Enum(data) => {
            let ty = input.ident;
            if data.variants.is_empty() {
                return syn::Error::new(ty.span(), "Enums with no variants are not supported")
                    .to_compile_error()
                    .into();
            }

            let has_fields = data
                .variants
                .iter()
                .any(|variant| !matches!(&variant.fields, syn::Fields::Unit));

            if has_fields {
                match full_enum_parser(input.attrs, data, ty) {
                    Ok(parser) => parser,
                    Err(err) => err.to_compile_error(),
                }
            } else {
                unit_enum_parser(data, ty)
            }
            .into()
        }
        _ => syn::Error::new(
            input.ident.span(),
            "Only structs and unit value enums are supported",
        )
        .to_compile_error()
        .into(),
    }
}

fn quote_fields(fields: Fields) -> TokenStream2 {
    match fields {
        Fields::Named(fields) => {
            let field_names = fields.named.iter().map(|f| f.ident.as_ref().unwrap());
            quote! {
                {
                    #(
                        #field_names
                    ),*
                }
            }
        }
        Fields::Unnamed(fields) => {
            let field_names = (0..fields.unnamed.len()).map(|i| format_ident!("data{}", i));
            quote! {
                (
                    #(
                        #field_names
                    ),*
                )
            }
        }
        Fields::Unit => {
            quote! {}
        }
    }
}

fn impl_unit_parser(attrs: &[syn::Attribute], ty: &Ident, construct: TokenStream2) -> TokenStream2 {
    let unit_parser = unit_parser(attrs, ty);
    quote! {
        impl kalosm_sample::Parse for #ty {
            fn new_parser() -> impl kalosm_sample::SendCreateParserState<Output = Self> {
                #unit_parser
                    .map_output(|_| #construct)
            }
        }
    }
}

fn unit_parse_literal(attrs: &[syn::Attribute], ty: &Ident) -> syn::Result<String> {
    // Look for #[parse(rename = "name")] attribute
    let mut ty_string = ty.unraw().to_string();
    for attr in attrs.iter() {
        if attr.path().is_ident("parse") {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("rename") {
                    let value = meta
                        .value()
                        .and_then(|value| value.parse::<syn::LitStr>())?;
                    ty_string = value.value();
                    Ok(())
                } else {
                    Err(meta.error("expected `rename`"))
                }
            })?;
        }
    }

    Ok(format!("\"{ty_string}\""))
}

fn unit_parser(attrs: &[syn::Attribute], ty: &Ident) -> TokenStream2 {
    let ty_string = match unit_parse_literal(attrs, ty) {
        Ok(ty_string) => ty_string,
        Err(err) => return err.to_compile_error(),
    };
    let ty_string = LitStr::new(&ty_string, ty.span());
    quote! {
        kalosm_sample::LiteralParser::new(#ty_string)
    }
}

fn full_enum_parser(
    attrs: Vec<syn::Attribute>,
    data: DataEnum,
    ty: Ident,
) -> syn::Result<TokenStream2> {
    // Look for the tag and content attributes within the #[parse] attribute
    let mut tag = "type".to_string();
    let mut content = "data".to_string();
    for attr in attrs.iter() {
        if attr.path().is_ident("parse") {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("tag") {
                    let value = meta
                        .value()
                        .and_then(|value| value.parse::<syn::LitStr>())?;
                    tag = value.value();
                    Ok(())
                } else if meta.path.is_ident("content") {
                    let value = meta
                        .value()
                        .and_then(|value| value.parse::<syn::LitStr>())?;
                    content = value.value();
                    Ok(())
                } else {
                    Err(meta.error("expected `tag` or `content`"))
                }
            })?;
        }
    }

    let mut parser = None;
    for variant in data.variants.iter() {
        let variant_ident = &variant.ident;
        let mut variant_name = variant_ident.unraw().to_string();
        // Look for #[parse(rename = "name")] attribute
        for attr in variant.attrs.iter() {
            if attr.path().is_ident("parse") {
                attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("rename") {
                        let value = meta
                            .value()
                            .and_then(|value| value.parse::<syn::LitStr>())?;
                        variant_name = value.value();
                        Ok(())
                    } else {
                        Err(meta.error("expected `rename`"))
                    }
                })?;
            }
        }

        let construct_variant = {
            let fields = quote_fields(variant.fields.clone());
            quote! {
                Self::#variant_ident #fields
            }
        };
        let parse_variant = match &variant.fields {
            syn::Fields::Named(fields) => {
                let parse_name_and_data = LitStr::new(
                    &format!("{}\", \"{content}\": ", variant_name),
                    variant.ident.span(),
                );
                let fields = fields.named.iter().collect::<Vec<_>>();
                let field_parser = field_parser(&fields, construct_variant)?;
                quote! {
                    kalosm_sample::LiteralParser::from(#parse_name_and_data).ignore_output_then(#field_parser)
                }
            }
            syn::Fields::Unnamed(fields) => {
                let field_vec = fields.unnamed.iter().collect::<Vec<_>>();
                let [inner] = *field_vec else {
                    return Err(syn::Error::new(
                        variant.ident.span(),
                        "Unnamed enum variants with more or less than one field are not supported",
                    ));
                };

                let parse_name_and_data = LitStr::new(
                    &format!("{}\", \"{content}\": ", variant_name),
                    variant.ident.span(),
                );
                let ty = &inner.ty;
                quote! {
                    kalosm_sample::LiteralParser::from(#parse_name_and_data).ignore_output_then(<#ty as kalosm_sample::Parse>::new_parser()).map_output(|data0| #construct_variant)
                }
            }
            // If this is a unit variant, we can just parse the type
            syn::Fields::Unit => {
                let lit_str_name =
                    LitStr::new(&format!("{}\"", variant_name), variant.ident.span());
                quote! {
                    kalosm_sample::LiteralParser::from(#lit_str_name).map_output(|_| #construct_variant)
                }
            }
        };
        match &mut parser {
            Some(current) => {
                *current = quote! {
                    #current
                        .or(
                            #parse_variant
                        )
                };
            }
            None => {
                parser = Some(parse_variant);
            }
        }
    }

    let struct_start = format!("{{ \"{tag}\": \"");

    Ok(quote! {
        impl kalosm_sample::Parse for #ty {
            fn new_parser() -> impl kalosm_sample::SendCreateParserState<Output = Self> {
                kalosm_sample::LiteralParser::from(#struct_start)
                    .ignore_output_then(#parser)
                    .then_literal(r#" }"#)
            }
        }
    })
}

fn unit_enum_parser(data: DataEnum, ty: Ident) -> TokenStream2 {
    // We can derive an efficient state machine for unit enums
    let parser_state = format_ident!("{}ParserState", ty);

    let mut parse_construction_map = HashMap::new();
    for variant in data.variants.iter() {
        let variant_name = &variant.ident;
        let fields = &variant.fields;
        let construct_variant = quote! {
            #ty::#variant_name #fields
        };
        let literal_string = match unit_parse_literal(&variant.attrs, variant_name) {
            Ok(literal_string) => literal_string,
            Err(err) => return err.to_compile_error(),
        };
        parse_construction_map.insert(literal_string.as_bytes().to_vec(), construct_variant);
    }

    let mut prefix_state_map = HashMap::new();
    let mut max_state = 0usize;
    for bytes in parse_construction_map.keys() {
        for i in 0..bytes.len() + 1 {
            let prefix = &bytes[..i];
            if prefix_state_map.contains_key(prefix) {
                continue;
            }
            prefix_state_map.insert(prefix, max_state);
            max_state += 1;
        }
    }

    let mut parse_states = Vec::new();
    for (state_prefix, state) in &prefix_state_map {
        let state = LitInt::new(&state.to_string(), ty.span());

        let mut next_bytes = Vec::new();
        for (next_state_prefix, next_state) in &prefix_state_map {
            if let Some(&[byte]) = next_state_prefix.strip_prefix(*state_prefix) {
                let next_state = LitInt::new(&next_state.to_string(), ty.span());

                next_bytes.push(quote! {
                    #byte => state = #parser_state(#next_state),
                });
            }
        }

        let unrecognized_byte = if let Some(constructor) = parse_construction_map.get(*state_prefix)
        {
            quote! {
                return kalosm_sample::ParseResult::Ok(kalosm_sample::ParseStatus::Finished {
                    result: #constructor,
                    remaining: &input[i..],
                })
            }
        } else {
            quote! {
                return kalosm_sample::ParseResult::Err(kalosm_sample::ParserError::msg("Unrecognized byte"))
            }
        };

        if !next_bytes.is_empty() {
            parse_states.push(quote! {
                #state => match byte {
                    #(#next_bytes)*
                    _ => #unrecognized_byte,
                },
            });
        } else if parse_construction_map.contains_key(*state_prefix) {
            parse_states.push(quote! {
                #state => #unrecognized_byte,
            });
        }
    }

    let mut match_required_next = Vec::new();
    for (state_prefix, state) in &prefix_state_map {
        let state = LitInt::new(&state.to_string(), ty.span());
        let mut required_next = Vec::new();
        let mut current_prefix = state_prefix.to_vec();
        loop {
            let mut valid_next_bytes = Vec::new();
            for prefix in prefix_state_map.keys() {
                if let Some(&[byte]) = prefix.strip_prefix(current_prefix.as_slice()) {
                    valid_next_bytes.push(byte);
                }
            }
            if let [byte] = *valid_next_bytes {
                current_prefix.push(byte);
                required_next.push(byte);
            } else {
                break;
            }
        }
        if !required_next.is_empty() {
            let required_next_str = String::from_utf8_lossy(&required_next);
            match_required_next.push(quote! {
                #state => #required_next_str,
            });
        }
    }

    let state_type = if max_state <= u8::MAX as usize {
        quote! { u8 }
    } else if max_state <= u16::MAX as usize {
        quote! { u16 }
    } else if max_state <= u32::MAX as usize {
        quote! { u32 }
    } else if max_state <= u64::MAX as usize {
        quote! { u64 }
    } else {
        quote! { u128 }
    };

    let impl_parser_state = quote! {
        #[derive(Debug, Clone, Copy)]
        struct #parser_state(#state_type);

        impl #parser_state {
            const fn new() -> Self {
                Self(0)
            }
        }
    };

    let parser = format_ident!("{}Parser", ty);
    let impl_parser = quote! {
        struct #parser;

        impl kalosm_sample::CreateParserState for #parser {
            fn create_parser_state(&self) -> <Self as kalosm_sample::Parser>::PartialState {
                #parser_state::new()
            }
        }
        impl kalosm_sample::Parser for #parser {
            type Output = #ty;
            type PartialState = #parser_state;

            fn parse<'a>(
                &self,
                state: &Self::PartialState,
                input: &'a [u8],
            ) -> kalosm_sample::ParseResult<kalosm_sample::ParseStatus<'a, Self::PartialState, Self::Output>> {
                let mut state = *state;
                for (i, byte) in input.iter().enumerate() {
                    match state.0 {
                        #(#parse_states)*
                        _ => return kalosm_sample::ParseResult::Err(kalosm_sample::ParserError::msg("Invalid state")),
                    }
                }
                kalosm_sample::ParseResult::Ok(kalosm_sample::ParseStatus::Incomplete {
                    new_state: state,
                    required_next: std::borrow::Cow::Borrowed(match state.0 {
                        #(#match_required_next)*
                        _ => ""
                    }),
                })
            }
        }
    };

    quote! {
        impl kalosm_sample::Parse for #ty {
            fn new_parser() -> impl kalosm_sample::SendCreateParserState<Output = Self> {
                #impl_parser_state
                #impl_parser

                #parser
            }
        }
    }
}

fn wrap_tuple(ident: &Ident, current: TokenStream2) -> TokenStream2 {
    quote! {
        (#current, #ident)
    }
}

fn field_parser(fields: &[&Field], construct: TokenStream2) -> syn::Result<TokenStream2> {
    let mut parsers = Vec::new();
    let idents: Vec<_> = fields
        .iter()
        .map(|f| format_ident!("{}_parser", f.ident.as_ref().unwrap().unraw()))
        .collect();
    for (i, (field, parser_ident)) in fields.iter().zip(idents.iter()).enumerate() {
        let ident = field.ident.as_ref().unwrap().unraw();

        let mut field_name = field.ident.as_ref().unwrap().unraw().to_string();
        let mut field_parser = {
            let ty = &field.ty;
            quote! {<#ty as kalosm_sample::Parse>::new_parser()}
        };
        // Look for #[parse(rename = "name")] or #[parse(with = expr)] attributes
        for attr in field.attrs.iter() {
            if attr.path().is_ident("parse") {
                attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("rename") {
                        let value = meta
                            .value()
                            .and_then(|value| value.parse::<syn::LitStr>())?;
                        field_name = value.value();
                        Ok(())
                    } else if meta.path.is_ident("with") {
                        let value = meta.value().and_then(|value| value.parse::<syn::Expr>())?;
                        field_parser = value.into_token_stream();
                        Ok(())
                    } else {
                        Err(meta.error("expected `rename` or `with`"))
                    }
                })?;
            }
        }

        let mut literal_text = String::new();
        if i == 0 {
            literal_text.push_str("{ ");
        } else {
            literal_text.push_str(", ");
        }
        literal_text.push_str(&format!("\"{field_name}\": "));
        let literal_text = LitStr::new(&literal_text, ident.span());

        parsers.push(quote! {
            let #parser_ident = kalosm_sample::LiteralParser::from(#literal_text)
                .ignore_output_then(#field_parser);
        });
    }

    let mut output_tuple = None;
    for field in fields.iter() {
        let name = field.ident.as_ref().unwrap();
        match output_tuple {
            Some(current) => {
                output_tuple = Some(wrap_tuple(name, current));
            }
            None => {
                output_tuple = Some(name.to_token_stream());
            }
        }
    }

    let mut join_parser: Option<TokenStream2> = None;
    for ident in idents.iter() {
        match &mut join_parser {
            Some(current) => {
                *current = quote! {
                    #current
                        .then(#ident)
                };
            }
            None => {
                join_parser = Some(ident.to_token_stream());
            }
        }
    }

    Ok(quote! {
        {
            #(
                #parsers
            )*

            #join_parser
                .then_literal(r#" }"#)
                .map_output(|#output_tuple| #construct)
        }
    })
}
