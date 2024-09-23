# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Provides the implementation for the EG-X model."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ._utils import _check_valid_array, _validate_float
from .base_hamiltonian import BaseHamiltonian


class EGXHamiltonian(BaseHamiltonian):
    """Hamiltonian for the EG-X model."""

    def __init__(
        self,
        hopping_gr: float,
        hopping_x: float,
        hopping_x_gr_a: float,
        lattice_constant: float,
        chemical_potential: float,
        hubbard_int_gr: float,
        hubbard_int_x: float,
        delta: npt.NDArray[np.complex64] | None = None,
        *args: tuple[Any, ...],
        **kwargs: tuple[dict[str, Any], ...],
    ) -> None:
        del args
        del kwargs
        self.hopping_gr = _validate_float(hopping_gr, "Hopping graphene")
        self.hopping_x = _validate_float(hopping_x, "Hopping impurity")
        self.hopping_x_gr_a = _validate_float(hopping_x_gr_a, "Hybridisation")
        self.lattice_constant = _validate_float(lattice_constant, "Lattice constant")
        self.chemical_potential = _validate_float(chemical_potential, "Chemical potential")
        self.hubbard_int_gr = _validate_float(hubbard_int_gr, "hubbard_int interaction graphene")
        self.hubbard_int_x = _validate_float(hubbard_int_x, "hubbard_int interaction impurity")
        self._hubbard_int_orbital_basis = np.array(
            [self.hubbard_int_gr, self.hubbard_int_gr, self.hubbard_int_x]
        )
        self._number_of_bands = 3
        if delta is None:
            self._delta_orbital_basis = np.zeros(3, dtype=np.complex64)
        else:
            self._delta_orbital_basis = delta

    @property
    def hubbard_int_orbital_basis(self) -> npt.NDArray[np.float64]:  # noqa: D102
        return self._hubbard_int_orbital_basis

    @property
    def delta_orbital_basis(self) -> npt.NDArray[np.complex64]:  # noqa: D102
        return self._delta_orbital_basis

    @delta_orbital_basis.setter
    def delta_orbital_basis(self, new_delta: npt.NDArray[np.complex64]) -> None:
        self._delta_orbital_basis = new_delta

    @property
    def number_of_bands(self) -> int:  # noqa: D102
        return self._number_of_bands

    def hamiltonian(self, k: npt.NDArray[np.float64]) -> npt.NDArray[np.complex64]:
        """
        Return the normal state Hamiltonian in orbital basis.

        Parameters
        ----------
        k : :class:`numpy.ndarray`
            List of k points.

        Returns
        -------
        :class:`numpy.ndarray`
            Hamiltonian in matrix form.

        """
        assert _check_valid_array(k)

        t_gr = self.hopping_gr
        t_x = self.hopping_x
        a = self.lattice_constant
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

    def hamiltonian_derivative(
        self, k: npt.NDArray[np.float64], direction: str
    ) -> npt.NDArray[np.complex64]:
        """
        Deriative of the Hamiltonian.

        Parameters
        ----------
        k: :class:`numpy.ndarray`
            List of k points.
        direction: str
            Direction for derivative, either 'x' oder 'y'.

        Returns
        -------
        :class:`numpy.ndarray`
            Derivative of Hamiltonian.

        """
        assert _check_valid_array(k)
        assert direction in ["x", "y"]

        t_gr = self.hopping_gr
        t_x = self.hopping_x
        a = self.lattice_constant
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
