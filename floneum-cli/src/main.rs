use std::path::PathBuf;

use clap::{Parser, Subcommand};
use floneum_plugin::*;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Build {
        #[arg(short, long)]
        features: Vec<String>,
        #[arg(short, long)]
        release: bool,
    },
    Package {
        #[arg(short, long)]
        features: Vec<String>,
        #[arg(short, long)]
        release: bool,
    },
}

#[tokio::main]
async fn main() {
    let args = Cli::parse();

    match args.command {
        Commands::Build { features, release } => {
            build(features, release);
        }
        Commands::Package { features, release } => {
            build(features, release);
            let build_path = {
                if release {
                    "target/wasm32-wasi/release"
                } else {
                    "target/wasm32-wasi/debug"
                }
            };
            let manafest = cargo_manifest::Manifest::from_path("Cargo.toml").unwrap();
            let mut plugin_manager = PluginEngine::default();
            let mut build_path = std::path::PathBuf::from(build_path);
            build_path = build_path
                .join(manafest.package.unwrap().name)
                .with_extension("wasm");
            let plugin = plugin_manager.load_plugin(&build_path).await;
            let instance = plugin.instance().await;
            let info = instance.metadata();
            let name = &info.name;
            let version = "0.1";
            let binding_version = "0.1";
            let description = &info.description;
            let package =
                floneumite::PackageStructure::new(name, version, description, binding_version);

            let package_path: PathBuf = "bundled".into();
            std::fs::create_dir_all(&package_path).unwrap();
            std::fs::write(
                package_path.join("floneum.toml"),
                toml::to_string(&package).unwrap(),
            )
            .unwrap();

            let wasm_path = package_path.join("package.wasm");
            std::fs::copy(&build_path, wasm_path).unwrap();
        }
    }
}

fn build(features: Vec<String>, release: bool) {
    let mut args: Vec<&str> = vec!["build", "--target", "wasm32-wasi"];
    if release {
        args.push("--release");
    }
    let features = features.join(",");
    if !features.is_empty() {
        args.push("--features");
        args.push(&features);
    }
    let status = std::process::Command::new("cargo")
        .args(&args)
        .status()
        .expect("failed to build plugin");
    if status.success() {
        println!("Build successful!");
        if release {
            println!("Build result placed in target/wasm32-wasi/release");
        } else {
            println!("Build result placed in target/wasm32-wasi/debug");
        }
    } else {
        println!("Build failed!");
    }
}
