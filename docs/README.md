# Building the Documentation

Floneum uses a fork of MdBook with multilanguage support. To build the documentation, you will need to install the forked version of MdBook.

```sh
cargo install mdbook --git https://github.com/Demonthos/mdBook.git --branch master
```

Then, you can build the documentation by running:

```sh
cd docs
mdbook build -d ./nightly
```
