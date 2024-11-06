.. SPDX-FileCopyrightText: 2024 Tjark Sievers
..
.. SPDX-License-Identifier: MIT

Development
===========

Contributing
------------

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
