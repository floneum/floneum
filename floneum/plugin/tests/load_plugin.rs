use std::path::Path;

use floneum_plugin::load_plugin;

#[tokio::test]
async fn load_plugin_works() {
    let root = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let crate_dir: &Path = root.as_ref();
    let path = crate_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("dist")
        .join("add")
        .join("package.wasm");
    println!("path: {:?}", path);
    let module = load_plugin(&path);
    let description = module.description().await.unwrap();

    println!("description: {:?}", description);
}
