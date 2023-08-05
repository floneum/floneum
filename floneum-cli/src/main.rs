use std::path::{Path, PathBuf};

use cargo_metadata::{Metadata, MetadataCommand};
use clap::{Parser, Subcommand};
use floneum_plugin::*;
use floneumite::{Config, PackageStructure};

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
        #[arg(short, long, value_delimiter = ',')]
        packages: Vec<String>,
        #[arg(short, long)]
        release: bool,
    },
}

#[tokio::main]
async fn main() {
    let args = Cli::parse();

    match args.command {
        Commands::Build { release, packages } => {
            build(release, packages, Some(&PathBuf::from("dist"))).await;
        }
    }
}

async fn build(release: bool, packages: Vec<String>, into: Option<&Path>) {
    let workspace = MetadataCommand::new().no_deps().exec().unwrap();

    let mut manafests = Config::default();

    if packages.is_empty() {
        if let Some(manifest) = package_and_build(None, release, &workspace, into).await {
            manafests.push(manifest);
        }
    } else {
        for package in &packages {
            if let Some(manifest) =
                package_and_build(Some(package), release, &workspace, into).await
            {
                manafests.push(manifest)
            }
        }
    }

    if let Some(into) = into {
        std::fs::create_dir_all(into).unwrap();
        std::fs::write(
            into.join("floneum.toml"),
            toml::to_string(&manafests).unwrap(),
        )
        .unwrap();
    }
}

async fn package_and_build(
    package: Option<&str>,
    release: bool,
    workspace: &Metadata,
    into: Option<&Path>,
) -> Option<PackageStructure> {
    build_package(release, package);
    let this_package = match package {
        Some(package) => workspace
            .packages
            .iter()
            .find(|pkg| &pkg.name == package)
            .unwrap_or_else(|| {
                panic!(
                    "package {} not found in {:?}",
                    package,
                    workspace
                        .packages
                        .iter()
                        .map(|pkg| &pkg.name)
                        .collect::<Vec<_>>()
                )
            }),
        None => workspace.root_package().expect("no root package found"),
    };

    let build_path = {
        let relative_to_root = if release {
            "wasm32-wasi/release"
        } else {
            "wasm32-wasi/debug"
        };
        let build_path = std::path::PathBuf::from(workspace.target_directory.as_str());
        build_path.join(relative_to_root)
    };
    if let Some(package_path) = into {
        let mut build_path = std::path::PathBuf::from(build_path);
        build_path = build_path
            .join(&this_package.name.replace('-', "_"))
            .with_extension("wasm");
        let plugin = load_plugin(&build_path);
        let instance = plugin.instance().await.unwrap();
        let info = instance.metadata();
        let name = &info.name;
        let version = this_package.version.to_string();
        let binding_version = this_package
            .dependencies
            .iter()
            .find(|dep| dep.name == "floneum_rust")
            .map(|dep| dep.req.to_string())
            .unwrap_or_else(|| "*".to_string());
        let description = &info.description;
        let authors = this_package.authors.clone();
        let package =
            floneumite::PackageStructure::new(name, &version, description, &binding_version)
                .with_authors(authors);

        let package_path = package_path.join(name);
        std::fs::create_dir_all(&package_path).unwrap();

        let wasm_path = package_path.join("package.wasm");
        std::fs::copy(&build_path, wasm_path).unwrap();

        Some(package)
    } else {
        None
    }
}

fn build_package(release: bool, package: Option<&str>) {
    let mut args: Vec<&str> = vec!["build", "--target", "wasm32-wasi"];
    if let Some(package) = package {
        args.push("--package");
        args.push(package);
    }
    if release {
        args.push("--release");
    }
    let mut command = std::process::Command::new("cargo");
    let command = command.args(&args);
    println!("building with: {:?}", command);
    let std::process::Output {
        stdout,
        stderr,
        status,
    } = command.output().expect("failed to build plugin");

    if status.success() {
        println!("Build successful!");
    } else {
        println!("Build failed!");
        println!("stdout: {}", String::from_utf8(stdout).unwrap());
        println!("stderr: {}", String::from_utf8(stderr).unwrap());
    }
}
