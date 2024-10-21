# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Command line interface."""

import sys
from typing import TextIO

import click
import yaml

from quant_met.parameters import Parameters

from .scf import scf


@click.command()
@click.argument("input-file", type=click.File("r"))
def cli(input_file: TextIO) -> None:
    """Command line interface for quant-met.

    Parameters
    ----------
    input_file
    """
    params = Parameters(**yaml.safe_load(input_file))

    match params.control.calculation:
        case "scf":
            scf(params)
        case _:
            print(f"Calculation {params.control.calculation} not found.")
            sys.exit(1)
