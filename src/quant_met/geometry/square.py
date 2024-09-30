# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Lattice geometry for Square Lattice."""

import numpy as np
import numpy.typing as npt

from .base_lattice import BaseLattice


class SquareLattice(BaseLattice):
    """Lattice geometry for Square Lattice."""

    def __init__(self, lattice_constant: np.float64) -> None:
        self._lattice_constant = lattice_constant
        self._bz_corners = (
            np.pi
            / lattice_constant
            * np.array([np.array([1, 1]), np.array([-1, 1]), np.array([1, -1]), np.array([-1, -1])])
        )
        self.Gamma = np.array([0, 0])
        self.M = np.pi / lattice_constant * np.array([1, 1])
        self.X = np.pi / lattice_constant * np.array([1, 0])
        self._high_symmetry_points = ((self.Gamma, r"\Gamma"), (self.M, "M"))

    @property
    def lattice_constant(self) -> np.float64:  # noqa: D102
        return self._lattice_constant

    @property
    def bz_corners(self) -> npt.NDArray[np.float64]:  # noqa: D102
        return self._bz_corners

    @property
    def high_symmetry_points(self) -> tuple[tuple[npt.NDArray[np.float64], str], ...]:  # noqa: D102
        return self._high_symmetry_points
