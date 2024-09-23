<!--
SPDX-FileCopyrightText: 2024 Tjark Sievers

SPDX-License-Identifier: MIT
-->

# quant-met

[![Test](https://github.com/Ruberhauptmann/quant-met/actions/workflows/test.yml/badge.svg)](https://github.com/Ruberhauptmann/quant-met/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/Ruberhauptmann/quant-met/badge.svg?branch=main)](https://coveralls.io/github/Ruberhauptmann/quant-met?branch=main)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/quant-met)](https://pypi.org/project/quant-met/)
[![PyPI - Version](https://img.shields.io/pypi/v/quant-met)](https://pypi.org/project/quant-met/)

This is a python package to treat superconductivity in flat-band systems.

* Documentation: [quant-met.tjarksievers.de](https://quant-met.tjarksievers.de)

## Installation

The package can be installed via
```shell
pip install quant-met
```

## Usage

For usage examples see [documentation](https://quant-met.tjarksievers.de/en/latest/examples.html).

## Contributing

This is a personal project, very geared to the work I did in my master's thesis.
If someone is using this and experiencing bugs or want the software extended, feel free to open an issue!

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

You can manually run tests using for example `tox -e py312` (for running against python 3.12).
After pushing your branch, all tests will also be run via Github Actions.

Using `pre-commit`, automatic linting and formatting is done before every commit, which may cause the first commit to fail.
A second try should then succeed.

To fix the reuse copyright:
```bash
  reuse annotate --license=MIT --copyright="Tjark Sievers" --skip-unrecognised -r .
```

After you are done working on an issue and all tests are running successful, you can add a new piece of changelog via `scriv create` and make a pull request.
