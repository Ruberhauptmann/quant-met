# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""
Utility functions (:mod:`quant_met.utils`)
==========================================

.. currentmodule:: quant_met.utils

Functions
---------

.. autosummary::
   :toctree: generated/

    generate_uniform_grid
"""  # noqa: D205, D400

import numpy as np
import numpy.typing as npt
from numba import jit


def generate_uniform_grid(
    ncols: int,
    nrows: int,
    corner_1: npt.NDArray[np.floating],
    corner_2: npt.NDArray[np.floating],
    origin: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Generate a uniform grid of points in 2D.

    Parameters
    ----------
        ncols : int
            Number of columns
        nrows : int
            Number of rows
        corner_1 : :py:class:`numpy.ndarray`
            First corner vector
        corner_2 : :py:class:`numpy.ndarray`
            Second corner vector
        origin : :py:class:`numpy.ndarray`
            Origin point

    Returns
    -------
        :py:class:`numpy.ndarray`
            Grid

    """
    if ncols <= 1 or nrows <= 1:
        msg = "Number of columns and rows must be greater than 1."
        raise ValueError(msg)
    if np.linalg.norm(corner_1) == 0 or np.linalg.norm(corner_2) == 0:
        msg = "Vectors to the corners cannot be zero."
        raise ValueError(msg)

    grid: npt.NDArray[np.floating] = np.concatenate(
        [
            np.linspace(
                origin[0] + i / (nrows - 1) * corner_2,
                origin[1] + corner_1 + i / (nrows - 1) * corner_2,
                num=ncols,
            )
            for i in range(nrows)
        ]
    )

    return grid


@jit
def fermi_dirac(energy: npt.NDArray[np.floating], beta: float) -> npt.NDArray[np.floating]:
    """Fermi dirac distribution.

    Parameters
    ----------
    energy
    beta

    Returns
    -------
    fermi_dirac

    """
    return (
        np.where(energy < 0, 1.0, 0.0)
        if np.isinf(beta)
        else np.asarray(1 / (1 + np.exp(beta * energy)))
    )
