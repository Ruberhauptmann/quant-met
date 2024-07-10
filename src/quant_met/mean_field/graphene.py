# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Provides the implementation for Graphene."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ._utils import _check_valid_array, _validate_float
from .base_hamiltonian import BaseHamiltonian


class GrapheneHamiltonian(BaseHamiltonian):
    """Hamiltonian for Graphene."""

    def __init__(
        self,
        t_nn: float,
        a: float,
        mu: float,
        coulomb_gr: float,
        delta: npt.NDArray[np.float64] | None = None,
        *args: tuple[Any, ...],
        **kwargs: tuple[dict[str, Any], ...],
    ) -> None:
        del args
        del kwargs
        self.t_nn = _validate_float(t_nn, "Hopping")
        if a <= 0:
            msg = "Lattice constant must be positive"
            raise ValueError(msg)
        self.a = _validate_float(a, "Lattice constant")
        self.mu = _validate_float(mu, "Chemical potential")
        self.coulomb_gr = _validate_float(coulomb_gr, "Coloumb interaction")
        self._coloumb_orbital_basis = np.array([self.coulomb_gr, self.coulomb_gr])
        self._number_of_bands = 2
        if delta is None:
            self._delta_orbital_basis = np.zeros(2)
        else:
            self._delta_orbital_basis = delta

    @property
    def number_of_bands(self) -> int:  # noqa: D102
        return self._number_of_bands

    @property
    def coloumb_orbital_basis(self) -> npt.NDArray[np.float64]:  # noqa: D102
        return self._coloumb_orbital_basis

    @property
    def delta_orbital_basis(self) -> npt.NDArray[np.float64]:  # noqa: D102
        return self._delta_orbital_basis

    @delta_orbital_basis.setter
    def delta_orbital_basis(self, new_delta: npt.NDArray[np.float64]) -> None:
        self._delta_orbital_basis = new_delta

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
        t_nn = self.t_nn
        a = self.a
        mu = self.mu
        if k.ndim == 1:
            k = np.expand_dims(k, axis=0)

        h = np.zeros((k.shape[0], self.number_of_bands, self.number_of_bands), dtype=np.complex64)

        h[:, 0, 1] = -t_nn * (
            np.exp(1j * k[:, 1] * a / np.sqrt(3))
            + 2 * np.exp(-0.5j * a / np.sqrt(3) * k[:, 1]) * (np.cos(0.5 * a * k[:, 0]))
        )
        h[:, 1, 0] = h[:, 0, 1].conjugate()
        h[:, 0, 0] -= mu
        h[:, 1, 1] -= mu

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

        t_nn = self.t_nn
        a = self.a
        if k.ndim == 1:
            k = np.expand_dims(k, axis=0)

        h = np.zeros((k.shape[0], self.number_of_bands, self.number_of_bands), dtype=np.complex64)

        if direction == "x":
            h[:, 0, 1] = (
                t_nn * a * np.exp(-0.5j * a / np.sqrt(3) * k[:, 1]) * np.sin(0.5 * a * k[:, 0])
            )
            h[:, 1, 0] = h[:, 0, 1].conjugate()
        else:
            h[:, 0, 1] = (
                -t_nn
                * 1j
                * a
                / np.sqrt(3)
                * (
                    np.exp(1j * a / np.sqrt(3) * k[:, 1])
                    - np.exp(-0.5j * a / np.sqrt(3) * k[:, 1]) * np.cos(0.5 * a * k[:, 0])
                )
            )
            h[:, 1, 0] = h[:, 0, 1].conjugate()

        return h.squeeze()
