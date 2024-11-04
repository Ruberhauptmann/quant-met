# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Provides the implementation for a three band tight binding model."""

import numpy as np
import numpy.typing as npt

from quant_met.geometry import SquareLattice
from quant_met.mean_field._utils import _check_valid_array
from quant_met.parameters import ThreeBandParameters

from .base_hamiltonian import BaseHamiltonian


class ThreeBand(BaseHamiltonian[ThreeBandParameters]):
    """Hamiltonian for Graphene."""

    def __init__(self, parameters: ThreeBandParameters) -> None:
        super().__init__(parameters)
        self.hopping = parameters.hopping
        self.chemical_potential = parameters.chemical_potential
        if parameters.delta is not None:
            self.delta_orbital_basis = np.astype(parameters.delta, np.complex64)

    def setup_lattice(self, parameters: ThreeBandParameters) -> SquareLattice:  # noqa: D102
        return SquareLattice(lattice_constant=parameters.lattice_constant)

    @classmethod
    def get_parameters_model(cls) -> type[ThreeBandParameters]:  # noqa: D102
        return ThreeBandParameters

    def hamiltonian(self, k: npt.NDArray[np.float64]) -> npt.NDArray[np.complex64]:  # noqa: D102
        assert _check_valid_array(k)
        hopping = self.hopping
        lattice_constant = self.lattice.lattice_constant
        chemical_potential = self.chemical_potential
        if k.ndim == 1:
            k = np.expand_dims(k, axis=0)

        h = np.zeros((k.shape[0], self.number_of_bands, self.number_of_bands), dtype=np.complex64)

        h[:, 0, 0] = (
            -2 * hopping * (np.cos(k[:, 1] * lattice_constant) + np.cos(k[:, 0] * lattice_constant))
        )
        h[:, 1, 1] = (
            -2 * hopping * (np.cos(k[:, 1] * lattice_constant) + np.cos(k[:, 0] * lattice_constant))
        )
        h[:, 2, 2] = (
            -2 * hopping * (np.cos(k[:, 1] * lattice_constant) + np.cos(k[:, 0] * lattice_constant))
        )
        h[:, 2, 0] = 0.5
        h[:, 0, 2] = 0.5

        h[:, 0, 0] -= chemical_potential
        h[:, 1, 1] -= chemical_potential
        h[:, 2, 2] -= chemical_potential

        return h.squeeze()

    def hamiltonian_derivative(  # noqa: D102
        self, k: npt.NDArray[np.float64], direction: str
    ) -> npt.NDArray[np.complex64]:
        assert _check_valid_array(k)
        assert direction in ["x", "y"]

        hopping = self.hopping
        lattice_constant = self.lattice.lattice_constant
        if k.ndim == 1:
            k = np.expand_dims(k, axis=0)

        h = np.zeros((k.shape[0], self.number_of_bands, self.number_of_bands), dtype=np.complex64)

        if direction == "x":
            h[:, 0, 0] = -2 * hopping * lattice_constant * np.sin(lattice_constant * k[:, 0])
            h[:, 1, 1] = -2 * hopping * lattice_constant * np.sin(lattice_constant * k[:, 0])
            h[:, 2, 2] = -2 * hopping * lattice_constant * np.sin(lattice_constant * k[:, 0])
        else:
            h[:, 0, 0] = -2 * hopping * lattice_constant * np.sin(lattice_constant * k[:, 0])
            h[:, 1, 1] = -2 * hopping * lattice_constant * np.sin(lattice_constant * k[:, 0])
            h[:, 2, 2] = -2 * hopping * lattice_constant * np.sin(lattice_constant * k[:, 0])

        return h.squeeze()
