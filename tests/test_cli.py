# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Tests for the command line interface."""

import os
from pathlib import Path

import yaml
from click.testing import CliRunner
from quant_met.cli import cli


def test_scf() -> None:
    """Test scf calculation via cli."""
    runner = CliRunner()
    parameters = {
        "model": {
            "name": "DressedGraphene",
            "hopping_gr": 1,
            "hopping_x": 0.01,
            "hopping_x_gr_a": 1,
            "chemical_potential": 0.0,
            "hubbard_int_gr": 0,
            "hubbard_int_x": 0,
            "lattice_constant": 3,
        },
        "control": {
            "calculation": "scf",
            "prefix": "test",
            "outdir": "test/",
            "conv_treshold": 1e-6,
        },
        "k_points": {"nk1": 30, "nk2": 30},
    }
    with runner.isolated_filesystem() and Path("input.yaml").open("w") as f:
        yaml.dump(parameters, f)

    result = runner.invoke(cli, ["input.yaml"])
    assert result.exit_code == 0
    Path("input.yaml").unlink()
    Path("test/test.hdf5").unlink()
    os.removedirs("test")


def test_scf_no_valid_calcution() -> None:
    """Test invalid calculation."""
    runner = CliRunner()
    parameters = {
        "model": {
            "name": "DressedGraphene",
            "hopping_gr": 1,
            "hopping_x": 0.01,
            "hopping_x_gr_a": 1,
            "chemical_potential": 0.0,
            "hubbard_int_gr": 1,
            "hubbard_int_x": 1,
            "lattice_constant": 3,
        },
        "control": {
            "calculation": "non-existent",
            "prefix": "test",
            "outdir": "test/",
            "beta": 100,
            "conv_treshold": 1e-6,
        },
        "k_points": {"nk1": 30, "nk2": 30},
    }
    with runner.isolated_filesystem() and Path("input.yaml").open("w") as f:
        yaml.dump(parameters, f)

    result = runner.invoke(cli, ["input.yaml"])
    assert result.exit_code == 1
    Path("input.yaml").unlink()
