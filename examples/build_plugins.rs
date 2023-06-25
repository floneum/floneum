use std::{fs::read_dir, path::PathBuf};

use floneum_plugin::PluginEngine;

#[tokio::main]
async fn main() {
    let plugins = [
        "add_embedding",
        "embedding",
        "embedding_db",
        "format",
        "infer",
        "infer_structured",
        "search",
    ];
    // build the plugins
    for plugin in plugins {
        let path = PathBuf::from("./plugins").join(plugin);
        let status = std::process::Command::new("cargo")
            .args(["build", "--target", "wasm32-wasi", "--release"])
            .current_dir(path)
            .status()
            .expect("failed to build plugin");
        assert!(status.success());
    }
    // publish the plugins
    let repo_path = PathBuf::from("../floneum-packages");
    std::fs::create_dir_all(&repo_path).unwrap();
    // remove the old packages
    for plugin in read_dir(&repo_path).unwrap().flatten() {
        let path = plugin.path();
        if path.is_dir()
            && path.file_name().unwrap() != "README.md"
            && !path.file_name().unwrap().to_string_lossy().contains("git")
        {
            std::fs::remove_dir_all(path).unwrap();
        }
    }
    let build_path = "target/wasm32-wasi/release";
    let mut plugin_manager = PluginEngine::default();
    for plugin in read_dir(build_path).unwrap().flatten() {
        let path = plugin.path();
        if path.extension() != Some(std::ffi::OsStr::new("wasm")) {
            continue;
        }
        let status = std::process::Command::new("wasm-tools")
            .args([
                "component",
                "new",
                &*path.to_string_lossy(),
                "-o",
                "my-component.wasm",
                "--adapt",
                "./plugin/wasi_snapshot_preview1.wasm",
            ])
            .status()
            .expect("failed to build plugin");
        assert!(status.success());
        let plugin = plugin_manager.load_plugin(&path).await;
        let instance = plugin.instance().await;
        let info = instance.metadata();
        let name = &info.name;
        let version = "0.1";
        let description = &info.description;
        let package = floneumate::PackageStructure::new(name, version, description);

        let package_path = repo_path.join(name);
        std::fs::create_dir_all(&package_path).unwrap();
        std::fs::write(
            package_path.join("floneum.toml"),
            toml::to_string(&package).unwrap(),
        )
        .unwrap();

        let wasm_path = package_path.join("package.wasm");
        std::fs::copy(path, wasm_path).unwrap();
    }
}
