# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Integration tests for the critical temperature subcommand."""

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from quant_met.cli import cli


@pytest.mark.slow_integration_test
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
            "n_temp_points": 5,
        },
        "k_points": {"nk1": 30, "nk2": 30},
    }
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with Path("input.yaml").open("w") as f:
            yaml.dump(parameters, f)
        result = runner.invoke(cli, ["input.yaml"])
        assert result.exit_code == 0


@pytest.mark.slow_integration_test
def test_crit_temp_small_number_of_points(tmp_path: Path) -> None:
    """Test crit_temp calculation via cli."""
    runner = CliRunner()
    parameters = {
        "model": {
            "name": "TwoBand",
            "hopping": 1,
            "chemical_potential": 0.0,
            "hubbard_int_orbital_basis": [1.0, 1.0],
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
