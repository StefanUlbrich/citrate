# Welcome to the Citrate workspace

This workspace includes the crates that I developed to learn Rust and
to support my research. It is named after the salts of the citric acid
and the design and development process and experience will be documented on
my [personal blog at lemonfold.io](https://lemonfold.io).

## The Potpourri Crate

A crate for models that can be trained by
[Expectation Maximization](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)
with a focus on [Mixture Models](https://en.wikipedia.org/wiki/Mixture_model).
Its  generic, flexible design that extensively uses the composition patterns is agnostic
of external dependencies (i.e., the numerical library chosen) and users only need
to implement a trait with few methods.

The crate is still heavily work in progress and currently, only clustering with [Gaussian Mixture
Models](https://scikit-learn.org/stable/modules/mixture.html) is in a rudimentary working state.
Future development, may include classification and regression with Mixture models, support for
more densities (Poisson, negative Binomial, van Mises, ..),
[SOM](https://en.wikipedia.org/wiki/Self-organizing_map),
[$k$-means](https://en.wikipedia.org/wiki/K-means_clustering),
[HMM](https://en.wikipedia.org/wiki/Hidden_Markov_model), or
[Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter).

To run the algorithm itself, you currently need to call a unit test:

```sh
RUST_BACKTRACE=full cargo test -F ndarray --package potpourri --lib -- mixture::tests::em_step --exact --nocapture
```

Currently, there are only models implemented for the [NDArrary](https://docs.rs/ndarray/latest/ndarray/) backend
but clusters and GPGPU is planned.

## The Cerebral Crate

A crate for self-organizing neural networks written in Rust. The crate is still work in progress
and not suited for use in other projects yet.

This is an experiment on writing a Python extension for numerical computation /
machine learning in Rust with Python bindings. The aim of this project is to
create an implementation of the [self-organizing
maps](https://en.wikipedia.org/wiki/Self-organizing_map) algorithm with support
for parallelization and eventually, GPGPU. The algorithm has been chosen for its
efficiency and simplicity (both in terms of implementation and
comprehensibility). Plus, it's my favorite and I'm doing active research with
it.

* Numerical python extensions written in Rust
* Seamless integration via [PyO3](https://pyo3.rs/v0.16.1/) and [rust-numpy](https://docs.rs/numpy/0.7.0/numpy/)
* High-performance via [ndarray](https://github.com/rust-ndarray/ndarray)/[rayon](https://docs.rs/ndarray/0.13.1/ndarray/parallel/index.html), [tch-rs](https://github.com/LaurentMazare/tch-rs) ([PyTorch](https://pytorch.org/) bindings)
* Dependency management / publishing with [Poetry](https://python-poetry.org/docs/) and [Maturin](https://github.com/PyO3/maturin)
* [Monorepo](https://en.wikipedia.org/wiki/Monorepo) with [Cargo workspaces][(](<https://doc.rust-lang.org/book/ch14-03-cargo-workspaces.html>))
* Technical documentation / GitHub page with [Sphinx](https://www.sphinx-doc.org/en/master/) and [MyST](https://myst-parser.readthedocs.io/en/latest/sphinx/intro.html)
* Eventually, distributed computation (e.g., [actor model](https://en.wikipedia.org/wiki/Actor_model) or [timely
  dataflow](https://timelydataflow.github.io/timely-dataflow/))
* Hopefully, GUI application using [Tauri](https://tauri.studio/), [Angular](https://angular.io/) and [ThreeJS](https://threejs.org/)

Feel free use as a basis for your own projects (MIT licensed).

## Development

For the monorepo, there is a top-level python project defined using poetry. This project is
mainly for dependency locking, integration testing and the overall project's documentation.
The python module
build with Maturin is in a nested folder and it's project file is also created with poetry.
This is necessary as the top-level project can only add local packages that are either
created with Poetry or contain a `setup.py` file.

While this setup supports a monorepo setup (and should support integration
testing on the imported local packages), there is another caveat. Building the python extension with
`pip` creates a temp directory which does not copy the local rust dependencies. Newer versions of `pip`
build within the tree, so this limitation can be avoided easily. Use the following commands to
setup the environment.

> :warning: We want to avoid creating a virtual environment for the nested packages. Work in
> a top-level shell instead.

```sh
cd self-organization
pyenv shell 3.10.2
poetry env use 3.10.2
poetry run pip install -U pip # only required until pip>=22.0 becomes the default
poetry install
# to debug / develop the extension
poetry shell
cd pysom
maturin develop
```

To install the virtual environment as a kernel for jupyter:

```sh
python -m ipykernel install --user --name py310_selforganization --display-name "Python3.10 (self-organization)"
```
