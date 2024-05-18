# quant-met


* Documentation:

## Installation

The package can be installed via

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

You can manually run tests using for example `tox -e py310` (for running against python 3.10).
After pushing your branch, all tests will also be run via Gitlab Actions.

Using `pre-commit`, automatic linting and formatting is done before every commit, which may cause the first commit to fail.
A second try should then succeed.

After you are done working on an issue and all tests are running successful, you can add a new piece of changelog via `scriv create` and make a merge request.
