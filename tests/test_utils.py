# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Test for the utility functions."""

import numpy as np
import pytest

from quant_met import utils


def test_generate_uniform_grid() -> None:
    """Test the function generate_uniform_grid."""
    grid = utils.generate_uniform_grid(
        ncols=3,
        nrows=3,
        corner_1=np.array([0, 1]),
        corner_2=np.array([1, 0]),
        origin=np.array([0, 0]),
    )
    np.testing.assert_array_equal(
        grid,
        np.array(
            [
                [0, 0],
                [0, 0.5],
                [0, 1],
                [0.5, 0],
                [0.5, 0.5],
                [0.5, 1],
                [1, 0],
                [1, 0.5],
                [1, 1],
            ]
        ),
        strict=True,
    )


def test_generate_uniform_grid_errors() -> None:
    """Test whether right errors are raised for generate_uniform_grid."""
    with pytest.raises(ValueError, match="Number of columns and rows must be greater than 1."):
        utils.generate_uniform_grid(
            ncols=1,
            nrows=10,
            corner_1=np.array([0, 1]),
            corner_2=np.array([1, 0]),
            origin=np.array([0, 0]),
        )
    with pytest.raises(ValueError, match="Number of columns and rows must be greater than 1."):
        utils.generate_uniform_grid(
            ncols=10,
            nrows=1,
            corner_1=np.array([0, 1]),
            corner_2=np.array([1, 0]),
            origin=np.array([0, 0]),
        )
    with pytest.raises(ValueError, match="Vectors to the corners cannot be zero."):
        utils.generate_uniform_grid(
            ncols=10,
            nrows=10,
            corner_1=np.array([0, 0]),
            corner_2=np.array([1, 0]),
            origin=np.array([0, 0]),
        )
    with pytest.raises(ValueError, match="Vectors to the corners cannot be zero."):
        utils.generate_uniform_grid(
            ncols=10,
            nrows=10,
            corner_1=np.array([1, 0]),
            corner_2=np.array([0, 0]),
            origin=np.array([0, 0]),
        )


def test_fermi_dirac() -> None:
    """Test the Fermi-Dirac distribution."""
    result = utils.fermi_dirac(np.array([-1, 0, 1]), 1)
    assert np.allclose(result, np.array([0.73105858, 0.5, 0.26894142]))
    result = utils.fermi_dirac(np.array([-1, 0, 1]), np.inf)
    assert np.all(result == np.array([1.0, 0.0, 0.0]))
