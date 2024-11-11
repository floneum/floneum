use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(Class)]
pub fn derive_class(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    let enum_data = if let syn::Data::Enum(e) = input.data {
        e
    } else {
        panic!("Class can only be derived for enums");
    };
    let ident = input.ident;

    let classes = enum_data.variants.len() as u32;
    let to_class = enum_data.variants.iter().enumerate().map(|(i, v)| {
        let ident = &v.ident;
        let i = i as u32;
        quote! {
            Self::#ident => #i,
        }
    });
    let from_class = enum_data.variants.iter().enumerate().map(|(i, v)| {
        let ident = &v.ident;
        let i = i as u32;
        quote! {
            #i => Self::#ident,
        }
    });

    let expanded = quote! {
        impl Class for #ident {
            const CLASSES: Option<u32> = Some(#classes);

            fn to_class(&self) -> u32 {
                match self {
                    #( #to_class )*
                }
            }
            fn from_class(class: u32) -> Self {
                match class {
                    #( #from_class )*
                    _ => panic!("Invalid class"),
                }
            }
        }
    };

    // Hand the output tokens back to the compiler
    TokenStream::from(expanded)
}
