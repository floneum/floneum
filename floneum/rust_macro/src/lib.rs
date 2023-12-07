use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};
use quote::{quote, ToTokens};
use syn::{
    parse::Parse, parse_macro_input, parse_quote, Error, FnArg, GenericArgument, ItemFn, LitStr,
    Meta, Path, PathArguments, PathSegment, ReturnType, Type,
};

mod inner {
    wit_bindgen::generate!({
        path: "../wit",
        world: "exports",

        exports: {
            world: MyHost,
        },
    });
}
use inner::plugins::main::types::{PrimitiveValueType, ValueType};

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
pub fn export_plugin(args: TokenStream, input: TokenStream) -> TokenStream {
    use convert_case::{Case, Casing};

    let mut input = parse_macro_input!(input as ItemFn);

    let function_ident = input.sig.ident.clone();
    let function_name = function_ident.to_string().to_case(Case::Title);
    let mut description = String::new();
    for attr in &input.attrs {
        if attr.path().is_ident("doc") {
            if let Meta::NameValue(meta) = &attr.meta {
                let value = &meta.value;
                let lit: LitStr = try_parse_quote!(#value);
                description += &lit.value().trim();
                description += "\n";
            }
        }
    }
    let mut examples: Option<proc_macro2::TokenStream> = None;
    let mut description = description.trim();
    if let Some((before, examples_code)) = description.split_once("### Examples") {
        examples = match examples_code.parse::<proc_macro2::TokenStream>() {
            Ok(examples) => Some(quote! {
                {
                    #examples
                }.into()
            }),
            Err(err) => {
                return syn::Error::new(
                    Span::call_site(),
                    format!("Failed to parse examples. Examples must be valid rust code {err}"),
                )
                .into_compile_error()
                .into();
            }
        };
        description = before.trim();
    }
    let examples = examples.unwrap_or_else(|| quote! {Vec::new()});

    let mut input_names: Vec<String> = Vec::new();
    let mut input_idents: Vec<Ident> = Vec::new();
    let mut input_types: Vec<IoDefinitionType> = Vec::new();
    let mut extract_inputs = Vec::new();
    for i in input.sig.inputs.iter_mut() {
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
            extract_inputs.push(ty.extract(ident));
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
                match syn::parse::<syn::ExprTuple>(args) {
                    Ok(ty) => {
                        for item in &ty.elems {
                            if let Ok(lit_str) = syn::parse2::<syn::LitStr>(quote! {
                                #item
                            }) {
                                output_names.push(lit_str.value());
                            }
                        }
                    }
                    Err(_) => {
                        for _ in 0..tuple.elems.len() {
                            output_names.push(format!("output{}", output_names.len()))
                        }
                    }
                };
                for ty in tuple.elems.iter() {
                    let ty = try_parse_quote!(#ty);
                    output_types.push(ty);
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
        ::floneum_rust::bindgen!(Plugin);

        #input

        pub struct Plugin;

        impl Guest for Plugin {
            fn structure() -> Definition {
                Definition {
                    name: #function_name.to_string(),
                    description: #description.to_string(),
                    inputs: vec![
                        #(
                            IoDefinition {
                                name: #input_names.to_string(),
                                ty: #input_types,
                            },
                        )*
                    ],
                    outputs: vec![
                        #(
                            IoDefinition {
                                name: #output_names.to_string(),
                                ty: #output_types,
                            },
                        )*
                    ],
                    examples: #examples,
                }
            }


            fn run(input: Vec<Input>) -> Vec<Output> {
                let mut input = input.into_iter();
                let __inner_fn = #function_ident;
                #(
                    #extract_inputs
                )*

                __inner_fn(#(#input_idents,)*).into_return_values()
            }
        }
    })
}

struct IoDefinitionType {
    value_type: ValueType,
}

impl IoDefinitionType {
    fn extract(&self, ident: Ident) -> proc_macro2::TokenStream {
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
            PrimitiveValueType::File => quote! {
                PrimitiveValue::File(inner)
            },
            PrimitiveValueType::Folder => quote! {
                PrimitiveValue::Folder(inner)
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
            PrimitiveValueType::EmbeddingModel => quote! {
                PrimitiveValue::EmbeddingModel(inner)
            },
            PrimitiveValueType::ModelType => quote! {
                PrimitiveValue::ModelType(inner)
            },
            PrimitiveValueType::EmbeddingModelType => quote! {
                PrimitiveValue::EmbeddingModelType(inner)
            },
            PrimitiveValueType::Boolean => quote! {
                PrimitiveValue::Boolean(inner)
            },
            PrimitiveValueType::Page => quote! {
                PrimitiveValue::Page(inner)
            },
            PrimitiveValueType::Node => quote! {
                PrimitiveValue::Node(inner)
            },
            PrimitiveValueType::Any => quote! {
                inner
            },
        };
        let quote = match &self.value_type {
            ValueType::Single(_) => {
                quote! {
                    Input::Single(#match_inner)
                }
            }
            ValueType::Many(_) => {
                quote! {
                    Input::Many(inner)
                }
            }
        };
        let get_return_value = match &self.value_type {
            ValueType::Single(_) => {
                quote! {
                    inner.into()
                }
            }
            ValueType::Many(_) => {
                quote! {
                    inner.iter().map(|inner| match inner {
                        #match_inner => inner.into(),
                        _ => panic!("unexpected input type {:?}", inner),
                    }).collect()
                }
            }
        };
        quote! {
            let __value = input.next().unwrap();
            let #ident = match __value {
                #quote => #get_return_value,
                _ => panic!("unexpected input type {:?}", __value),
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
                PrimitiveValueType::Number
            },
            PrimitiveValueType::Text => quote! {
                PrimitiveValueType::Text
            },
            PrimitiveValueType::File => quote! {
                PrimitiveValueType::File
            },
            PrimitiveValueType::Folder => quote! {
                PrimitiveValueType::Folder
            },
            PrimitiveValueType::Embedding => quote! {
                PrimitiveValueType::Embedding
            },
            PrimitiveValueType::Database => quote! {
                PrimitiveValueType::Database
            },
            PrimitiveValueType::Model => quote! {
                PrimitiveValueType::Model
            },
            PrimitiveValueType::EmbeddingModel => quote! {
                PrimitiveValueType::EmbeddingModel
            },
            PrimitiveValueType::ModelType => quote! {
                PrimitiveValueType::ModelType
            },
            PrimitiveValueType::EmbeddingModelType => quote! {
                PrimitiveValueType::EmbeddingModelType
            },
            PrimitiveValueType::Boolean => quote! {
                PrimitiveValueType::Boolean
            },
            PrimitiveValueType::Page => quote! {
                PrimitiveValueType::Page
            },
            PrimitiveValueType::Node => quote! {
                PrimitiveValueType::Node
            },
            PrimitiveValueType::Any => quote! {
                PrimitiveValueType::Any
            },
        };
        let quote = match &self.value_type {
            ValueType::Single(_) => quote! {
                ValueType::Single(#quote_inner)
            },
            ValueType::Many(_) => quote! {
                ValueType::Many(#quote_inner)
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
        let mut ident = segments.next().unwrap();

        if ident.ident == "Option" {
            if let PathArguments::AngleBracketed(inner) = ident.arguments {
                if let Some(GenericArgument::Type(Type::Path(item_type))) = inner.args.iter().next()
                {
                    let path = &item_type.path;
                    let inner: PathSegment = path.segments.first().unwrap().clone();
                    ident = inner.clone();
                } else {
                    return Err(Error::new_spanned(
                        inner,
                        "Option must have a simple type as it's Generic".to_string(),
                    ));
                }
            } else {
                return Err(Error::new_spanned(
                    ident,
                    "Option missing Generics".to_string(),
                ));
            }
        }

        let primitive_type = if ident.ident == "Vec" {
            many = true;
            if let PathArguments::AngleBracketed(inner) = ident.arguments {
                if let Some(GenericArgument::Type(Type::Path(item_type))) = inner.args.iter().next()
                {
                    let Some(ident) = item_type.path.get_ident() else {
                        return Err(Error::new_spanned(
                            item_type,
                            "Vec missing Generics".to_string(),
                        ));
                    };
                    parse_primitive_value_type(ident)?
                } else {
                    return Err(Error::new_spanned(
                        inner,
                        "Vec must have a simple type as it's Generic".to_string(),
                    ));
                }
            } else {
                return Err(Error::new_spanned(
                    ident,
                    "Vec missing Generics".to_string(),
                ));
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
    } else if ident == "Model" {
        Ok(PrimitiveValueType::Model)
    } else if ident == "EmbeddingModel" {
        Ok(PrimitiveValueType::EmbeddingModel)
    } else if ident == "EmbeddingDb" {
        Ok(PrimitiveValueType::Database)
    } else if ident == "Embedding" {
        Ok(PrimitiveValueType::Embedding)
    } else if ident == "ModelType" {
        Ok(PrimitiveValueType::ModelType)
    } else if ident == "EmbeddingModelType" {
        Ok(PrimitiveValueType::EmbeddingModelType)
    } else if ident == "bool" {
        Ok(PrimitiveValueType::Boolean)
    } else if ident == "File" {
        Ok(PrimitiveValueType::File)
    } else if ident == "Folder" {
        Ok(PrimitiveValueType::Folder)
    } else if ident == "PrimitiveValue" {
        Ok(PrimitiveValueType::Any)
    } else if ident == "Page" || ident == "PageId" {
        Ok(PrimitiveValueType::Page)
    } else if ident == "Node" {
        Ok(PrimitiveValueType::Node)
    } else {
        let error = format!("type {} not allowed. Inputs and outputs must be one of i64, String, ModelInstance, EmbeddingDatabase, Embedding, ModelType, bool, PrimitiveValue, Page, Node", ident);
        Err(Error::new_spanned(ident, error))
    }
}
