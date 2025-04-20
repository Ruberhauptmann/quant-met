# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Command line interface."""

import logging
import sys
from typing import TextIO

import click
import yaml

from quant_met.parameters import Parameters

from .crit_temp import crit_temp
from .scf import scf

logger = logging.getLogger(__name__)


@click.command()
@click.argument("input-file", type=click.File("r"))
@click.option("--debug", is_flag=True, help="Enable debug logging.")
def cli(input_file: TextIO, *, debug: bool) -> None:
    """Command line interface for quant-met.

    Parameters
    ----------
    input_file: TextIO
        A file object containing YAML formatted parameters for the simulation.
    debug : bool
        If set, enables debug logging instead of the default info logging.

    This command reads the parameters from the specified file and runs the
    desired calculation based on the provided parameters.
    """
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stdout,
        )
        logger.setLevel(logging.DEBUG)
        logger.info("Debug logging is enabled.")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stdout,
        )

    params = Parameters(**yaml.safe_load(input_file))
    logger.info("Loaded parameters successfully.")

    match params.control.calculation:
        case "scf":
            logger.info("Starting SCF calculation.")
            scf(params)
        case "crit-temp":
            logger.info("Starting T_C calculation.")
            crit_temp(params)
        case _:
            logger.error("Calculation %s not found.", params.control.calculation)
            sys.exit(1)
