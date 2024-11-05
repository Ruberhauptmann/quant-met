# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Provides the implementation for the dressed graphene model."""

import numpy as np
import numpy.typing as npt

from quant_met.geometry import BaseLattice
from quant_met.geometry.graphene import GrapheneLattice
from quant_met.mean_field._utils import _check_valid_array
from quant_met.parameters.hamiltonians import DressedGrapheneParameters

from .base_hamiltonian import BaseHamiltonian


class DressedGraphene(BaseHamiltonian[DressedGrapheneParameters]):
    """Hamiltonian for the dressed graphene model."""

    def __init__(self, parameters: DressedGrapheneParameters) -> None:
        super().__init__(parameters)
        self.hopping_gr = parameters.hopping_gr
        self.hopping_x = parameters.hopping_x
        self.hopping_x_gr_a = parameters.hopping_x_gr_a
        self.hubbard_int_orbital_basis = parameters.hubbard_int_orbital_basis
        self.chemical_potential = parameters.chemical_potential
        if parameters.delta is not None:
            self.delta_orbital_basis = np.astype(parameters.delta, np.complex64)

    def setup_lattice(self, parameters: DressedGrapheneParameters) -> BaseLattice:  # noqa: D102
        return GrapheneLattice(lattice_constant=parameters.lattice_constant)

    @classmethod
    def get_parameters_model(cls) -> type[DressedGrapheneParameters]:  # noqa: D102
        return DressedGrapheneParameters

    def hamiltonian(self, k: npt.NDArray[np.float64]) -> npt.NDArray[np.complex64]:  # noqa: D102
        assert _check_valid_array(k)

        t_gr = self.hopping_gr
        t_x = self.hopping_x
        a = self.lattice.lattice_constant
        v = self.hopping_x_gr_a
        chemical_potential = self.chemical_potential
        if k.ndim == 1:
            k = np.expand_dims(k, axis=0)

        h = np.zeros((k.shape[0], self.number_of_bands, self.number_of_bands), dtype=np.complex64)

        h[:, 0, 1] = -t_gr * (
            np.exp(1j * k[:, 1] * a / np.sqrt(3))
            + 2 * np.exp(-0.5j * a / np.sqrt(3) * k[:, 1]) * (np.cos(0.5 * a * k[:, 0]))
        )

        h[:, 1, 0] = h[:, 0, 1].conjugate()

        h[:, 2, 0] = v
        h[:, 0, 2] = v

        h[:, 2, 2] = (
            -2
            * t_x
            * (
                np.cos(a * k[:, 0])
                + 2 * np.cos(0.5 * a * k[:, 0]) * np.cos(0.5 * np.sqrt(3) * a * k[:, 1])
            )
        )
        h[:, 0, 0] -= chemical_potential
        h[:, 1, 1] -= chemical_potential
        h[:, 2, 2] -= chemical_potential

        return h.squeeze()

    def hamiltonian_derivative(  # noqa: D102
        self, k: npt.NDArray[np.float64], direction: str
    ) -> npt.NDArray[np.complex64]:
        assert _check_valid_array(k)
        assert direction in ["x", "y"]

        t_gr = self.hopping_gr
        t_x = self.hopping_x
        a = self.lattice.lattice_constant
        if k.ndim == 1:
            k = np.expand_dims(k, axis=0)

        h = np.zeros((k.shape[0], self.number_of_bands, self.number_of_bands), dtype=np.complex64)

        if direction == "x":
            h[:, 0, 1] = (
                t_gr * a * np.exp(-0.5j * a / np.sqrt(3) * k[:, 1]) * np.sin(0.5 * a * k[:, 0])
            )
            h[:, 1, 0] = h[:, 0, 1].conjugate()
            h[:, 2, 2] = (
                2
                * a
                * t_x
                * (
                    np.sin(a * k[:, 0])
                    + np.sin(0.5 * a * k[:, 0]) * np.cos(0.5 * np.sqrt(3) * a * k[:, 1])
                )
            )
        else:
            h[:, 0, 1] = (
                -t_gr
                * 1j
                * a
                / np.sqrt(3)
                * (
                    np.exp(1j * a / np.sqrt(3) * k[:, 1])
                    - np.exp(-0.5j * a / np.sqrt(3) * k[:, 1]) * np.cos(0.5 * a * k[:, 0])
                )
            )
            h[:, 1, 0] = h[:, 0, 1].conjugate()
            h[:, 2, 2] = np.sqrt(3) * a * t_x * np.cos(0.5 * np.sqrt(3) * a * k[:, 1])

        return h.squeeze()
