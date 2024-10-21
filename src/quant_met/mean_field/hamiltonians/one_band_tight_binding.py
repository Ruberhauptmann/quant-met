# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Provides the implementation for Graphene."""

import pathlib
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from quant_met.geometry import SquareLattice
from quant_met.mean_field._utils import _check_valid_array, _validate_float
from quant_met.parameters import OneBandParameters

from .base_hamiltonian import BaseHamiltonian


class OneBand(BaseHamiltonian):
    """Hamiltonian for Graphene."""

    def __init__(
        self,
        parameters: OneBandParameters,
        *args: tuple[Any, ...],
        **kwargs: tuple[dict[str, Any], ...],
    ) -> None:
        del args
        del kwargs
        self._name = parameters.name
        self.hopping = _validate_float(parameters.hopping, "Hopping")
        if parameters.lattice_constant <= 0:
            msg = "Lattice constant must be positive"
            raise ValueError(msg)
        self._lattice = SquareLattice(
            np.float64(_validate_float(parameters.lattice_constant, "Lattice constant"))
        )
        self.lattice_constant = self._lattice.lattice_constant
        self.chemical_potential = _validate_float(
            parameters.chemical_potential, "Chemical potential"
        )
        self.hubbard_int = _validate_float(parameters.hubbard_int, "Hubbard interaction")
        self._hubbard_int_orbital_basis = np.array([self.hubbard_int])
        self._number_of_bands = 1
        if parameters.delta is None:
            self._delta_orbital_basis = np.zeros(self.number_of_bands, dtype=np.complex64)
        else:
            self._delta_orbital_basis = np.astype(parameters.delta, np.complex64)

    @property
    def name(self) -> str:  # noqa: D102
        return self._name

    @property
    def lattice(self) -> SquareLattice:  # noqa: D102
        return self._lattice

    @property
    def number_of_bands(self) -> int:  # noqa: D102
        return self._number_of_bands

    @property
    def hubbard_int_orbital_basis(self) -> npt.NDArray[np.float64]:  # noqa: D102
        return self._hubbard_int_orbital_basis

    @property
    def delta_orbital_basis(self) -> npt.NDArray[np.complex64]:  # noqa: D102
        return self._delta_orbital_basis

    @delta_orbital_basis.setter
    def delta_orbital_basis(self, new_delta: npt.NDArray[np.complex64]) -> None:
        self._delta_orbital_basis = new_delta

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> "BaseHamiltonian":  # noqa: D102
        with h5py.File(f"{filename}", "r") as f:
            config_dict = dict(f.attrs.items())
            config_dict["delta"] = f["delta"][()]
        parameters = OneBandParameters.model_validate(config_dict)
        return cls(parameters=parameters)

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
        hopping = self.hopping
        lattice_constant = self.lattice.lattice_constant
        chemical_potential = self.chemical_potential
        if k.ndim == 1:
            k = np.expand_dims(k, axis=0)

        h = np.zeros((k.shape[0], self.number_of_bands, self.number_of_bands), dtype=np.complex64)

        h[:, 0, 0] = (
            -2 * hopping * (np.cos(k[:, 1] * lattice_constant) + np.cos(k[:, 0] * lattice_constant))
        )
        h[:, 0, 0] -= chemical_potential

        return h

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

        hopping = self.hopping
        lattice_constant = self.lattice.lattice_constant
        if k.ndim == 1:
            k = np.expand_dims(k, axis=0)

        h = np.zeros((k.shape[0], self.number_of_bands, self.number_of_bands), dtype=np.complex64)

        if direction == "x":
            h[:, 0, 0] = -2 * hopping * lattice_constant * np.sin(lattice_constant * k[:, 0])
        else:
            h[:, 0, 0] = -2 * hopping * lattice_constant * np.sin(lattice_constant * k[:, 0])

        return h.squeeze()
