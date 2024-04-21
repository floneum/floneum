pub use floneum_rust_macro::export_plugin;
mod helpers;
pub use helpers::*;

wit_bindgen::generate!({
    path: "../wit",
    world: "plugin-world",
    pub_export_macro: true,
});
