# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Tests for the command line interface."""

from pathlib import Path

import yaml
from click.testing import CliRunner
from quant_met.cli import cli


def test_scf(tmp_path: Path) -> None:
    """Test scf calculation via cli."""
    runner = CliRunner()
    parameters = {
        "model": {
            "name": "DressedGraphene",
            "hopping_gr": 1,
            "hopping_x": 0.01,
            "hopping_x_gr_a": 1,
            "chemical_potential": 0.0,
            "hubbard_int_orbital_basis": [0.0, 0.0, 0.0],
            "lattice_constant": 3,
        },
        "control": {
            "calculation": "scf",
            "prefix": "test",
            "outdir": "test/",
            "conv_treshold": 1e-2,
        },
        "k_points": {"nk1": 30, "nk2": 30},
    }
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with Path("input.yaml").open("w") as f:
            yaml.dump(parameters, f)
        result = runner.invoke(cli, ["input.yaml"])
        result = runner.invoke(cli, ["--debug", "input.yaml"])
        assert result.exit_code == 0


def test_no_valid_calculation(tmp_path: Path) -> None:
    """Test invalid calculation."""
    runner = CliRunner()
    parameters = {
        "model": {
            "name": "DressedGraphene",
            "hopping_gr": 1,
            "hopping_x": 0.01,
            "hopping_x_gr_a": 1,
            "chemical_potential": 0.0,
            "hubbard_int_orbital_basis": [0.0, 0.0, 0.0],
            "lattice_constant": 3,
        },
        "control": {
            "calculation": "non-existent",
            "prefix": "test",
            "outdir": "test/",
            "beta": 100,
            "conv_treshold": 1e-2,
        },
        "k_points": {"nk1": 30, "nk2": 30},
    }
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with Path("input.yaml").open("w") as f:
            yaml.dump(parameters, f)
        result = runner.invoke(cli, ["input.yaml"])
        assert result.exit_code == 1


def test_crit_temp(tmp_path: Path) -> None:
    """Test crit_temp calculation via cli."""
    runner = CliRunner()
    parameters = {
        "model": {
            "name": "OneBand",
            "hopping": 1,
            "chemical_potential": 0.0,
            "hubbard_int_orbital_basis": [1.0],
            "lattice_constant": 1,
        },
        "control": {
            "calculation": "crit-temp",
            "prefix": "test",
            "outdir": "test",
            "conv_treshold": 1e-3,
            "n_temp_points": 20,
        },
        "k_points": {"nk1": 30, "nk2": 30},
    }
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with Path("input.yaml").open("w") as f:
            yaml.dump(parameters, f)
        result = runner.invoke(cli, ["input.yaml"])
        assert result.exit_code == 0


def test_crit_temp_small_number_of_points(tmp_path: Path) -> None:
    """Test crit_temp calculation via cli."""
    runner = CliRunner()
    parameters = {
        "model": {
            "name": "DressedGraphene",
            "hopping_gr": 1,
            "hopping_x": 0.01,
            "hopping_x_gr_a": 2,
            "chemical_potential": 0.0,
            "hubbard_int_orbital_basis": [1.0, 1.0, 1.0],
            "lattice_constant": 1,
        },
        "control": {
            "calculation": "crit-temp",
            "prefix": "test",
            "outdir": "test",
            "conv_treshold": 1e-3,
            "n_temp_points": 1,
        },
        "k_points": {"nk1": 30, "nk2": 30},
    }
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with Path("input.yaml").open("w") as f:
            yaml.dump(parameters, f)
        result = runner.invoke(cli, ["input.yaml"])
        assert result.exit_code == 0
