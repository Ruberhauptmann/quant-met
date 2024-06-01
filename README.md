# quant-met

[![Test](https://github.com/Ruberhauptmann/quant-met/actions/workflows/test.yml/badge.svg)](https://github.com/Ruberhauptmann/quant-met/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/Ruberhauptmann/quant-met/badge.svg?branch=main)](https://coveralls.io/github/Ruberhauptmann/quant-met?branch=main)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/quant-met)
![PyPI - Version](https://img.shields.io/pypi/v/quant-met)


* Documentation: [quant-met.readthedocs.io](https://quant-met.readthedocs.io/en/latest/)

## Installation

The package can be installed via
```shell
pip install quant-met
```

## Usage

## Contributing

You are welcome to open an issue if you want something changed or added in the software or if there are bugs occuring.

### Developing

You can also help develop this software further.
This should help you get set up to start this.

Prerequisites:
* make
* python
* conda

Set up the development environment:
* clone the repository
* run `make environment`
* now activate the conda environment `conda activate quant-met-dev`

Now you can create a separate branch to work on the project.

You can manually run tests using for example `tox -e py312` (for running against python 3.12).
After pushing your branch, all tests will also be run via Gitlab Actions.

Using `pre-commit`, automatic linting and formatting is done before every commit, which may cause the first commit to fail.
A second try should then succeed.

After you are done working on an issue and all tests are running successful, you can add a new piece of changelog via `scriv create` and make a merge request.
