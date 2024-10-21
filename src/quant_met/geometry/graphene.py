# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Lattice geometry for Graphene."""

import numpy as np
import numpy.typing as npt

from .base_lattice import BaseLattice


class GrapheneLattice(BaseLattice):
    """Lattice geometry for Graphene."""

    def __init__(self, lattice_constant: np.float64) -> None:
        self._lattice_constant = lattice_constant
        self._bz_corners = (
            4
            * np.pi
            / (3 * self.lattice_constant)
            * np.array([(np.cos(i * np.pi / 3), np.sin(i * np.pi / 3)) for i in range(6)])
        )
        self.Gamma = np.array([0, 0])
        self.M = np.pi / self.lattice_constant * np.array([1, 1 / np.sqrt(3)])
        self.K = 4 * np.pi / (3 * self.lattice_constant) * np.array([1, 0])
        self._high_symmetry_points = ((self.M, "M"), (self.Gamma, r"\Gamma"), (self.K, "K"))

    @property
    def lattice_constant(self) -> np.float64:  # noqa: D102
        return self._lattice_constant

    @property
    def bz_corners(self) -> npt.NDArray[np.float64]:  # noqa: D102
        return self._bz_corners

    @property
    def high_symmetry_points(self) -> tuple[tuple[npt.NDArray[np.float64], str], ...]:  # noqa: D102
        return self._high_symmetry_points
