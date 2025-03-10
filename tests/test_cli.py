# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Tests for the command line interface."""

from pathlib import Path
from unittest.mock import MagicMock, Mock

import yaml
from click.testing import CliRunner
from pytest_mock import MockerFixture

from quant_met import mean_field, parameters
from quant_met.cli import cli


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


def test_crit_temp_mock(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test crit-temp calculation with mock."""
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
            "calculation": "crit-temp",
            "prefix": "test",
            "outdir": "test/",
            "beta": 100,
            "conv_treshold": 1e-2,
            "n_temp_points": 10,
        },
        "k_points": {"nk1": 30, "nk2": 30},
    }
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with Path("input.yaml").open("w") as f:
            yaml.dump(parameters, f)
        mock_search_crit_temp = mocker.patch("quant_met.mean_field.search_crit_temp")
        mock_search_crit_temp.return_value = (MagicMock(), [0.1, 0.2], MagicMock())
        result = runner.invoke(cli, ["input.yaml", "--debug"])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["input.yaml"])
        mean_field.search_crit_temp.assert_called()
        assert result.exit_code == 0


def test_scf_mock(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test scf calculation with mock."""
    runner = CliRunner()
    model_parameters = {
        "name": "DressedGraphene",
        "hopping_gr": 1,
        "hopping_x": 0.01,
        "hopping_x_gr_a": 1,
        "chemical_potential": 0.0,
        "hubbard_int_orbital_basis": [0.0, 0.0, 0.0],
        "lattice_constant": 3,
    }
    test_parameters = {
        "model": model_parameters,
        "control": {
            "calculation": "scf",
            "prefix": "test",
            "outdir": "test/",
            "beta": 100,
            "conv_treshold": 1e-2,
            "calculate_additional": True
        },
        "k_points": {"nk1": 30, "nk2": 30},
    }
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with Path("input.yaml").open("w") as f:
            yaml.dump(test_parameters, f)
        mock_search_self_consistency_loop = mocker.patch(
            "quant_met.mean_field.self_consistency_loop"
        )
        mock_search_self_consistency_loop.return_value = mean_field.hamiltonians.DressedGraphene(
            parameters=parameters.DressedGrapheneParameters(
                **model_parameters
            )
        )
        result = runner.invoke(cli, ["input.yaml", "--debug"])
        mean_field.self_consistency_loop.assert_called()
        assert result.exit_code == 0
    with runner.isolated_filesystem(temp_dir=tmp_path):
        test_parameters['control']['calculate_additional'] = False
        with Path("input.yaml").open("w") as f:
            yaml.dump(test_parameters, f)
        mock_search_self_consistency_loop = mocker.patch(
            "quant_met.mean_field.self_consistency_loop"
        )
        mock_search_self_consistency_loop.return_value = mean_field.hamiltonians.DressedGraphene(
            parameters=parameters.DressedGrapheneParameters(
                **model_parameters
            )
        )
        result = runner.invoke(cli, ["input.yaml"])
        mean_field.self_consistency_loop.assert_called()
        assert result.exit_code == 0
