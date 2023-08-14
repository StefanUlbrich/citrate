# Welcome to the `Cerebral` Python package


This package contains Python bindings for the
[Cerebral](https://stefanulbrich.github.io/citrate/cerebral/index.html) crate.

## Installation

For now, please install by cloning the repository or installing with the following
command as the package is not on PyPI yet.

```sh
pip install "git+https://github.com/StefanUlbrich/citrate.git#egg=cerebral&subdirectory=cerebral"
```

## Development

Clone this repository and install [Poetry](https://python-poetry.org/docs/#installation).
It is recommended to familiarize yourself with Poetry first.
Make sure you have a recent Python version in your path.
Then, create a virtual environment and install the bindings

```sh
cd cerebral
poetry install
```

If you want to install a jupyter lab environment, then install the `jupyter` dependency group

```sh
poetry install --with jupyter
```

If you have multiple python versions available, you can specify which one
to use (prior to the previous command)

```sh
poetry env use python3.10 # example
```

For development, you will have to build (and install) the project quite often.
The following command does this

```sh
poetry run maturin develop --release
```


You can build the wheel file with

```sh
poetry run maturin build --release
ls ../target/wheels/cerebral-*.whl
# e.g., ../target/wheels/cerebral-0.1.1-cp310-cp310-manylinux_2_34_x86_64.whl
```

Finally, you can clean up / remove the virtual environment with

```sh
poetry env remove --all
```
