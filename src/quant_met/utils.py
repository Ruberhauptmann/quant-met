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


def generate_uniform_grid(
    ncols: int,
    nrows: int,
    corner_1: npt.NDArray[np.float64],
    corner_2: npt.NDArray[np.float64],
    origin: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
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

    grid: npt.NDArray[np.float64] = np.concatenate(
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
