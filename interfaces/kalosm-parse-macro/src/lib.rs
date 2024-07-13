use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};
use syn::{ext::IdentExt, parse_macro_input, DeriveInput, Field, Ident, LitStr};
use syn::{DataEnum, Fields};

#[proc_macro_derive(Parse, attributes(parse))]
pub fn derive_parse(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    match input.data {
        syn::Data::Struct(data) => match data.fields {
            syn::Fields::Named(fields) => {
                let ty = input.ident;
                if fields.named.is_empty() {
                    return TokenStream::from(impl_unit_parser(&ty, quote! { Self {} }));
                }

                let field_names = fields.named.iter().map(|f| f.ident.as_ref().unwrap());
                let construct = quote! {
                    Self {
                        #(
                            #field_names
                        ),*
                    }
                };

                let parser = field_parser(&fields.named.iter().collect::<Vec<_>>(), construct);
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
                TokenStream::from(impl_unit_parser(&ty, quote! { Self }))
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
                full_enum_parser(data, ty)
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

fn impl_unit_parser(ty: &Ident, construct: TokenStream2) -> TokenStream2 {
    let unit_parser = unit_parser(ty);
    quote! {
        impl kalosm_sample::Parse for #ty {
            fn new_parser() -> impl kalosm_sample::SendCreateParserState<Output = Self> {
                #unit_parser
                    .map_output(|_| #construct)
            }
        }
    }
}

fn unit_parser(ty: &Ident) -> TokenStream2 {
    let ty_string = LitStr::new(&format!("\"{}\"", ty.unraw()), ty.span());
    quote! {
        kalosm_sample::LiteralParser::new(#ty_string)
    }
}

fn full_enum_parser(data: DataEnum, ty: Ident) -> TokenStream2 {
    let mut parser = None;
    for variant in data.variants.iter() {
        let variant_name = &variant.ident;

        let construct_variant = {
            let fields = quote_fields(variant.fields.clone());
            quote! {
                Self::#variant_name #fields
            }
        };
        let parse_variant = match &variant.fields {
            syn::Fields::Named(fields) => {
                let parse_name_and_data = LitStr::new(
                    &format!("{}\",\"data\":", variant_name.unraw()),
                    variant.ident.span(),
                );
                let fields = fields.named.iter().collect::<Vec<_>>();
                let field_parser = field_parser(&fields, construct_variant);
                quote! {
                    kalosm_sample::LiteralParser::from(#parse_name_and_data).ignore_output_then(#field_parser)
                }
            }
            syn::Fields::Unnamed(fields) => {
                let field_vec = fields.unnamed.iter().collect::<Vec<_>>();
                let [inner] = *field_vec else {
                    return syn::Error::new(
                        variant.ident.span(),
                        "Unnamed enum variants with more or less than one field are not supported",
                    )
                    .to_compile_error();
                };

                let parse_name_and_data = LitStr::new(
                    &format!("{}\",\"data\":", variant_name.unraw()),
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
                    LitStr::new(&format!("{}\"", variant_name.unraw()), variant.ident.span());
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

    quote! {
        impl kalosm_sample::Parse for #ty {
            fn new_parser() -> impl kalosm_sample::SendCreateParserState<Output = Self> {
                kalosm_sample::LiteralParser::from(r#"{"type":""#)
                    .ignore_output_then(#parser)
                    .then_literal(r#"}"#)
            }
        }
    }
}

fn unit_enum_parser(data: DataEnum, ty: Ident) -> TokenStream2 {
    let mut parser = None;
    for variant in data.variants.iter() {
        let variant_name = &variant.ident;
        let fields = &variant.fields;
        let lit_str_name = LitStr::new(
            &format!("\"{}\"", variant_name.unraw()),
            variant.ident.span(),
        );
        let construct_variant = quote! {
            Self::#variant_name #fields
        };
        let parse_variant = quote! {
            kalosm_sample::LiteralParser::from(#lit_str_name).map_output(|_| #construct_variant)
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

    quote! {
        impl kalosm_sample::Parse for #ty {
            fn new_parser() -> impl kalosm_sample::SendCreateParserState<Output = Self> {
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

fn field_parser(fields: &[&Field], construct: TokenStream2) -> TokenStream2 {
    let mut parsers = Vec::new();
    let idents: Vec<_> = fields
        .iter()
        .map(|f| format_ident!("{}_parser", f.ident.as_ref().unwrap().unraw()))
        .collect();
    for (i, (field, parser_ident)) in fields.iter().zip(idents.iter()).enumerate() {
        let ident = field.ident.as_ref().unwrap().unraw();
        let mut literal_text = String::new();
        if i == 0 {
            literal_text.push('{');
        } else {
            literal_text.push(',');
        }
        literal_text.push_str(&format!("\"{}\":", field.ident.as_ref().unwrap().unraw()));
        let literal_text = LitStr::new(&literal_text, ident.span());
        // Try to grab the parser from the `#[parse(with = expr)]` attribute, otherwise use the default parser
        let field_parser = if let Some(attr) = field
            .attrs
            .iter()
            .find(|attr| attr.path().is_ident("parse"))
        {
            let mut parser = None;
            let result = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("with") {
                    let value = meta.value().and_then(|value| value.parse::<syn::Expr>())?;
                    parser = Some(value.into_token_stream());
                    Ok(())
                } else {
                    Err(meta.error("expected `with`"))
                }
            });
            if let Err(err) = result {
                return err.to_compile_error();
            }
            parser.to_token_stream()
        } else {
            let ty = &field.ty;
            quote! {<#ty as kalosm_sample::Parse>::new_parser()}
        };

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

    quote! {
        {
            #(
                #parsers
            )*

            #join_parser
                .then_literal(r#"}"#)
                .map_output(|#output_tuple| #construct)
        }
    }
}
