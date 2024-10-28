# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Base class for lattice geometries."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from quant_met.utils import generate_uniform_grid

from .bz_path import generate_bz_path


class BaseLattice(ABC):
    """Base class for lattice geometries."""

    @property
    @abstractmethod
    def lattice_constant(self) -> float:  # pragma: no cover
        """Lattice constant."""
        raise NotImplementedError

    @property
    @abstractmethod
    def bz_corners(self) -> npt.NDArray[np.float64]:  # pragma: no cover
        """Corners of the BZ."""
        raise NotImplementedError

    @property
    @abstractmethod
    def reciprocal_basis(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:  # pragma: no cover
        """Reciprocal basis vectors."""
        raise NotImplementedError

    @property
    @abstractmethod
    def high_symmetry_points(
        self,
    ) -> tuple[tuple[npt.NDArray[np.float64], str], ...]:  # pragma: no cover
        """Tuple of high symmetry points and names."""
        raise NotImplementedError

    def generate_bz_grid(self, ncols: int, nrows: int) -> npt.NDArray[np.float64]:
        """Generate a grid in the BZ.

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
            ncols,
            nrows,
            self.reciprocal_basis[0],
            self.reciprocal_basis[1],
            origin=np.array([0, 0]),
        )

    def generate_high_symmetry_path(
        self, number_of_points: int
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        list[float],
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
