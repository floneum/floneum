use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};
use syn::{ext::IdentExt, parse_macro_input, DeriveInput, Ident, LitStr};

#[proc_macro_derive(Parse, attributes(parse))]
pub fn derive_parse(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    match input.data {
        syn::Data::Struct(data) => match data.fields {
            syn::Fields::Named(fields) => {
                let ty = input.ident;
                if fields.named.is_empty() {
                    return TokenStream::from(unit_parser(&ty));
                }
                let mut parsers = Vec::new();
                let idents: Vec<_> = fields
                    .named
                    .iter()
                    .map(|f| format_ident!("{}_parser", f.ident.as_ref().unwrap().unraw()))
                    .collect();
                for (i, (field, parser_ident)) in fields.named.iter().zip(idents.iter()).enumerate()
                {
                    let ident = field.ident.as_ref().unwrap().unraw();
                    let mut literal_text = String::new();
                    if i == 0 {
                        literal_text.push('{');
                    } else {
                        literal_text.push(',');
                    }
                    literal_text
                        .push_str(&format!("\"{}\":", field.ident.as_ref().unwrap().unraw()));
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
                                let value =
                                    meta.value().and_then(|value| value.parse::<syn::Expr>())?;
                                parser = Some(value.into_token_stream());
                                Ok(())
                            } else {
                                Err(meta.error("expected `with`"))
                            }
                        });
                        if let Err(err) = result {
                            return err.to_compile_error().into();
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

                let wrap_tuple = |ident: &Ident, current: TokenStream2| {
                    quote! {
                        (#current, #ident)
                    }
                };

                let mut output_tuple = None;
                for field in fields.named.iter() {
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

                let field_names = fields.named.iter().map(|f| f.ident.as_ref().unwrap());

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

                let parse_block = quote! {
                    #(
                        #parsers
                    )*

                    #join_parser
                        .then_literal(r#"}"#)
                        .map_output(|#output_tuple| Self {
                            #(
                                #field_names
                            ),*
                        })
                };

                let expanded = quote! {
                    impl kalosm_sample::Parse for #ty {
                        fn new_parser() -> impl kalosm_sample::SendCreateParserState<Output = Self> {
                            #parse_block
                        }
                    }
                };
                TokenStream::from(expanded)
            }
            syn::Fields::Unit => {
                let ty = input.ident;
                TokenStream::from(unit_parser(&ty))
            }
            _ => panic!("Only structs with named fields are supported"),
        },
        syn::Data::Enum(data) => {
            let ty = input.ident;
            if data.variants.is_empty() {
                return TokenStream::from(unit_parser(&ty));
            }

            let mut parser = None;
            for variant in data.variants.iter() {
                let variant_name = &variant.ident;
                let fields = &variant.fields;
                if !fields.is_empty() {
                    return TokenStream::from(
                        syn::Error::new(
                            variant.ident.span(),
                            "Enums with non-unit variants are not supported",
                        )
                        .to_compile_error(),
                    );
                }
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
            .into()
        }
        _ => panic!("Only structs and unit value enums are supported"),
    }
}

fn unit_parser(ty: &Ident) -> TokenStream2 {
    let ty_string = LitStr::new(&format!("\"{}\"", ty.unraw()), ty.span());
    quote! {
        impl kalosm_sample::Parse for #ty {
            fn new_parser() -> impl kalosm_sample::SendCreateParserState<Output = Self> {
                kalosm_sample::LiteralParser::new(#ty_string)
                    .map_output(|_| Self {})
            }
        }
    }
}
