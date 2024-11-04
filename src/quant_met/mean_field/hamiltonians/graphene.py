# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Provides the implementation for Graphene."""

import numpy as np
import numpy.typing as npt

from quant_met.geometry import GrapheneLattice
from quant_met.mean_field._utils import _check_valid_array
from quant_met.parameters.hamiltonians import GrapheneParameters

from .base_hamiltonian import BaseHamiltonian


class Graphene(BaseHamiltonian[GrapheneParameters]):
    """Hamiltonian for Graphene."""

    def __init__(
        self,
        parameters: GrapheneParameters,
    ) -> None:
        super().__init__(parameters)
        self.hopping = parameters.hopping
        self.chemical_potential = parameters.chemical_potential
        if parameters.delta is not None:
            self.delta_orbital_basis = np.astype(parameters.delta, np.complex64)

    def setup_lattice(self, parameters: GrapheneParameters) -> GrapheneLattice:  # noqa: D102
        return GrapheneLattice(lattice_constant=parameters.lattice_constant)

    @classmethod
    def get_parameters_model(cls) -> type[GrapheneParameters]:  # noqa: D102
        return GrapheneParameters

    def hamiltonian(self, k: npt.NDArray[np.float64]) -> npt.NDArray[np.complex64]:  # noqa: D102
        assert _check_valid_array(k)
        hopping = self.hopping
        lattice_constant = self.lattice.lattice_constant
        chemical_potential = self.chemical_potential
        if k.ndim == 1:
            k = np.expand_dims(k, axis=0)

        h = np.zeros((k.shape[0], self.number_of_bands, self.number_of_bands), dtype=np.complex64)

        h[:, 0, 1] = -hopping * (
            np.exp(1j * k[:, 1] * lattice_constant / np.sqrt(3))
            + 2
            * np.exp(-0.5j * lattice_constant / np.sqrt(3) * k[:, 1])
            * (np.cos(0.5 * lattice_constant * k[:, 0]))
        )
        h[:, 1, 0] = h[:, 0, 1].conjugate()
        h[:, 0, 0] -= chemical_potential
        h[:, 1, 1] -= chemical_potential

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
            h[:, 0, 1] = (
                hopping
                * lattice_constant
                * np.exp(-0.5j * lattice_constant / np.sqrt(3) * k[:, 1])
                * np.sin(0.5 * lattice_constant * k[:, 0])
            )
            h[:, 1, 0] = h[:, 0, 1].conjugate()
        else:
            h[:, 0, 1] = (
                -hopping
                * 1j
                * lattice_constant
                / np.sqrt(3)
                * (
                    np.exp(1j * lattice_constant / np.sqrt(3) * k[:, 1])
                    - np.exp(-0.5j * lattice_constant / np.sqrt(3) * k[:, 1])
                    * np.cos(0.5 * lattice_constant * k[:, 0])
                )
            )
            h[:, 1, 0] = h[:, 0, 1].conjugate()

        return h.squeeze()
