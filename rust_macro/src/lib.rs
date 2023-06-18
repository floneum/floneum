use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::{quote, ToTokens};
use syn::{
    parse::Parse, parse_macro_input, parse_quote, Error, FnArg, GenericArgument, ItemFn, LitStr,
    Meta, Path, PathArguments, ReturnType, Type,
};

#[allow(unused_macros)]
mod inner {
    wit_bindgen::generate!({path: "../wit"});
}
use inner::exports::plugins::main::definitions::{PrimitiveValueType, ValueType};

macro_rules! try_parse_quote {
    ($($tokens:tt)*) => {
        match syn::parse2(quote!($($tokens)*)) {
            Ok(ty) => ty,
            Err(err) => {
                return err.to_compile_error().into();
            }
        }
    };
}

#[proc_macro_attribute]
pub fn export_plugin(_: TokenStream, input: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(input as ItemFn);

    let function_ident = input.sig.ident.clone();
    let function_name = function_ident.to_string();
    let mut discription = String::new();
    for attr in &input.attrs {
        if attr.path().is_ident("doc") {
            if let Meta::NameValue(meta) = &attr.meta {
                let value = &meta.value;
                let lit: LitStr = try_parse_quote!(#value);
                discription += &lit.value();
                discription += "\n";
            }
        }
    }

    let mut input_names: Vec<String> = Vec::new();
    let mut input_idents: Vec<Ident> = Vec::new();
    let mut input_types: Vec<IoDefinitionType> = Vec::new();
    let mut extract_inputs = Vec::new();
    for (idx, i) in input.sig.inputs.iter_mut().enumerate() {
        if let FnArg::Typed(typed) = i {
            let ident: Ident = {
                let pat = &*typed.pat;
                try_parse_quote! {
                    #pat
                }
            };
            input_idents.push(ident.clone());
            let mut name = ident.to_string();
            typed.attrs.retain(|attr| {
                let is_discription = attr.path().is_ident("doc");
                if is_discription {
                    if let Meta::NameValue(meta) = &attr.meta {
                        let value = &meta.value;
                        let lit: LitStr = parse_quote!(#value);
                        name = lit.value();
                    }
                }
                !is_discription
            });
            input_names.push(name);
            let ty = &typed.ty;
            let ty: IoDefinitionType = try_parse_quote!(#ty);
            extract_inputs.push(ty.extract(ident, idx));
            input_types.push(ty);
        } else {
            return quote! {
                compiler_error!("self not allowed in inputs")
            }
            .into();
        }
    }

    let mut output_names: Vec<String> = Vec::new();
    let mut output_types: Vec<IoDefinitionType> = Vec::new();
    match &input.sig.output {
        ReturnType::Type(_, ty) => match &**ty {
            Type::Tuple(tuple) => {
                for ty in tuple.elems.iter() {
                    let ty = try_parse_quote!(#ty);
                    output_types.push(ty);
                    output_names.push(format!("output{}", output_names.len()))
                }
            }
            _ => {
                let ty = try_parse_quote!(#ty);
                output_types.push(ty);
                output_names.push("output".to_string())
            }
        },
        ReturnType::Default => {}
    }

    // Hand the resulting function body back to the compiler.
    TokenStream::from(quote! {
        #input

        rust_adapter::export_plugin_world!(Plugin);

        pub struct Plugin;

        impl rust_adapter::Definitions for Plugin {
            fn structure() -> rust_adapter::Definition {
                rust_adapter::Definition {
                    name: #function_name.to_string(),
                    description: #discription.to_string(),
                    inputs: vec![
                        #(
                            rust_adapter::IoDefinition {
                                name: #input_names.to_string(),
                                ty: #input_types,
                            },
                        )*
                    ],
                    outputs: vec![
                        #(
                            rust_adapter::IoDefinition {
                                name: #output_names.to_string(),
                                ty: #output_types,
                            },
                        )*
                    ],
                }
            }

            fn run(input: Vec<rust_adapter::Value>) -> Vec<rust_adapter::Value> {
                #(
                    #extract_inputs
                )*

                use rust_adapter::IntoReturnValues;

                #function_ident(#(#input_idents,)*).into_return_values()
            }
        }
    })
}

struct IoDefinitionType {
    value_type: ValueType,
}

impl IoDefinitionType {
    fn extract(&self, ident: Ident, idx: usize) -> proc_macro2::TokenStream {
        let inner = match &self.value_type {
            ValueType::Single(inner) => inner,
            ValueType::Many(inner) => inner,
        };
        let match_inner = match inner {
            PrimitiveValueType::Number => quote! {
                PrimitiveValue::Number(inner)
            },
            PrimitiveValueType::Text => quote! {
                PrimitiveValue::Text(inner)
            },
            PrimitiveValueType::Embedding => quote! {
                PrimitiveValue::Embedding(inner)
            },
            PrimitiveValueType::Database => quote! {
                PrimitiveValue::Database(inner)
            },
            PrimitiveValueType::Model => quote! {
                PrimitiveValue::Model(inner)
            },
        };
        let quote = match &self.value_type {
            ValueType::Single(_) => {
                quote! {
                    Value::Single(#match_inner)
                }
            }
            ValueType::Many(_) => {
                quote! {
                    Value::Many(#match_inner)
                }
            }
        };
        quote! {
            let value = &input[#idx];
            let #ident = match value {
                #quote => inner.clone(),
                _ => panic!("unexpected input type {:?}", value),
            };
        }
    }
}

impl ToTokens for IoDefinitionType {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let inner = match &self.value_type {
            ValueType::Single(inner) => inner,
            ValueType::Many(inner) => inner,
        };
        let quote_inner = match inner {
            PrimitiveValueType::Number => quote! {
                rust_adapter::PrimitiveValueType::Number
            },
            PrimitiveValueType::Text => quote! {
                rust_adapter::PrimitiveValueType::Text
            },
            PrimitiveValueType::Embedding => quote! {
                rust_adapter::PrimitiveValueType::Embedding
            },
            PrimitiveValueType::Database => quote! {
                rust_adapter::PrimitiveValueType::Database
            },
            PrimitiveValueType::Model => quote! {
                rust_adapter::PrimitiveValueType::Model
            },
        };
        let quote = match &self.value_type {
            ValueType::Single(_) => quote! {
                rust_adapter::ValueType::Single(#quote_inner)
            },
            ValueType::Many(_) => quote! {
                rust_adapter::ValueType::Many(#quote_inner)
            },
        };
        tokens.extend(quote)
    }
}

impl Parse for IoDefinitionType {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ty: Path = input.parse()?;
        let mut segments = ty.segments.into_iter();
        let mut many = false;
        let ident = segments.next().unwrap();
        let primitive_type = if ident.ident == "Vec" {
            many = true;
            if let PathArguments::AngleBracketed(inner) = ident.arguments {
                if let Some(GenericArgument::Type(Type::Path(item_type))) = inner.args.iter().next()
                {
                    let Some(ident) = item_type.path.get_ident()
                        else{
                            return Err(Error::new_spanned(item_type,format!(
                                "Vec missing Generics"
                            )))
                        };
                    parse_primitive_value_type(ident)?
                } else {
                    return Err(Error::new_spanned(
                        inner,
                        format!("Vec must have a simple type as a genric"),
                    ));
                }
            } else {
                return Err(Error::new_spanned(ident, format!("Vec missing Generics")));
            }
        } else {
            parse_primitive_value_type(&ident.ident)?
        };

        let value_type = if many {
            ValueType::Many(primitive_type)
        } else {
            ValueType::Single(primitive_type)
        };

        Ok(IoDefinitionType { value_type })
    }
}

fn parse_primitive_value_type(ident: &Ident) -> syn::Result<PrimitiveValueType> {
    if ident == "i64" {
        Ok(PrimitiveValueType::Number)
    } else if ident == "String" {
        Ok(PrimitiveValueType::Text)
    } else if ident == "ModelInstance" {
        Ok(PrimitiveValueType::Model)
    } else if ident == "EmbeddingDbId" {
        Ok(PrimitiveValueType::Database)
    } else if ident == "Embedding" {
        Ok(PrimitiveValueType::Embedding)
    } else {
        let error = format!("type {} not allowed. Inputs and outputs must be one of i64, String, ModelInstance, VectorDatabase", ident.to_token_stream().to_string());
        Err(Error::new_spanned(ident, error))
    }
}
