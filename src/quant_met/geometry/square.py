# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Lattice geometry for Square Lattice."""

from typing import Any

import numpy as np
import numpy.typing as npt
from numpy import dtype, generic, ndarray

from quant_met.utils import generate_uniform_grid

from .bz_path import generate_bz_path


class SquareLattice:
    """Lattice geometry for Square Lattice."""

    lattice_constant = 1
    bz_corners = (
        np.pi
        / lattice_constant
        * np.array([np.array([1, 1]), np.array([-1, 1]), np.array([1, -1]), np.array([-1, -1])])
    )

    Gamma = np.array([0, 0])
    M = np.pi / lattice_constant * np.array([1, 1])
    X = np.pi / lattice_constant * np.array([1, 0])

    high_symmetry_points = ((Gamma, r"\Gamma"), (M, "M"))

    def generate_bz_grid(self, ncols: int, nrows: int) -> npt.NDArray[np.float64]:
        """Generate a grid in the graphene BZ.

        Parameters
        ----------
        ncols : int
            Number of points in column.
        nrows : int
            Number of points in row.

        Returns
        -------
        :class:`numpy.ndarray`
            Array of grid points in the BZ.

        """
        return generate_uniform_grid(
            ncols, nrows, self.bz_corners[0], self.bz_corners[1], origin=np.array([0, 0])
        )

    def generate_high_symmetry_path(
        self, number_of_points: int
    ) -> tuple[
        ndarray[Any, dtype[generic | Any]],
        ndarray[Any, dtype[generic | Any]],
        list[int | Any],
        list[str],
    ]:
        """Generate a path through high symmetry points.

        Parameters
        ----------
        number_of_points: int
            Number of point in the whole path.

        Returns
        -------
        :class:`numpy.ndarray`
            List of two-dimensional k points.
        :class:`numpy.ndarray`
            Path for plotting purposes: points between 0 and 1, with appropriate spacing.
        list[float]
            A list of ticks for the plotting path.
        list[str]
            A list of labels for the plotting path.

        """
        return generate_bz_path(list(self.high_symmetry_points), number_of_points=number_of_points)
