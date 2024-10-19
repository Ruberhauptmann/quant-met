# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Command line interface."""

import pathlib

import click
import yaml

from quant_met.parameters import Parameters


@click.command()
@click.option("--input-file", type=click.Path(), help="Input file", required=True)
def cli(input_file: pathlib.Path) -> None:
    """Command line interface for quant-met.

    Parameters
    ----------
    input_file
    """
    with input_file.open() as f:
        params = Parameters(**yaml.safe_load(f))

    print(params.model)
