# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Provides the base class for Hamiltonians."""

import pathlib
from abc import ABC, abstractmethod

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

from ._utils import _check_valid_array


class BaseHamiltonian(ABC):
    """Base class for Hamiltonians."""

    @property
    @abstractmethod
    def number_of_bands(self) -> int:
        """Number of bands in the model."""
        raise NotImplementedError

    @property
    def coloumb_orbital_basis(self) -> npt.NDArray[np.float64]:
        """
        Coloumb interaction split up in orbitals.

        Returns
        -------
        :class:`numpy.ndarray`

        """
        raise NotImplementedError

    @property
    def delta_orbital_basis(self) -> npt.NDArray[np.complex64]:
        """
        Order parameter in orbital basis.

        Returns
        -------
        :class:`numpy.ndarray`

        """
        raise NotImplementedError

    @delta_orbital_basis.setter
    @abstractmethod
    def delta_orbital_basis(self, new_delta: npt.NDArray[np.complex64]) -> None:
        raise NotImplementedError

    @abstractmethod
    def _hamiltonian_one_point(self, k_point: npt.NDArray[np.float64]) -> npt.NDArray[np.complex64]:
        raise NotImplementedError

    @abstractmethod
    def _hamiltonian_derivative_one_point(
        self, k_point: npt.NDArray[np.float64], directions: str
    ) -> npt.NDArray[np.complex64]:
        raise NotImplementedError

    def _bdg_hamiltonian_one_point(
        self, k_point: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.complex64]:
        delta_matrix: npt.NDArray[np.complex64] = np.zeros(
            shape=(self.number_of_bands, self.number_of_bands), dtype=np.complex64
        )
        np.fill_diagonal(delta_matrix, self.delta_orbital_basis)

        return np.block(
            [
                [self.hamiltonian(k_point), delta_matrix],
                [np.conjugate(delta_matrix), -np.conjugate(self.hamiltonian(-k_point))],
            ]
        )

    def save(self, filename: pathlib.Path) -> None:
        """
        Save the Hamiltonian as a HDF5 file.

        Parameters
        ----------
        filename : :class:`pathlib.Path`
            Filename to save the Hamiltonian to, should end in .hdf5

        """
        with h5py.File(f"{filename}", "a") as f:
            f.create_dataset("delta", data=self.delta_orbital_basis)
            for key, value in vars(self).items():
                if not key.startswith("_"):
                    f.attrs[key] = value

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> "BaseHamiltonian":
        """
        Initialise a Hamiltonian from a HDF5 file.

        Parameters
        ----------
        filename : :class:`pathlib.Path`
            File to load the Hamiltonian from.

        """
        with h5py.File(f"{filename}", "r") as f:
            config_dict = dict(f.attrs.items())
            config_dict["delta"] = f["delta"][()]

        return cls(**config_dict)

    def bdg_hamiltonian(self, k: npt.NDArray[np.float64]) -> npt.NDArray[np.complex64]:
        """
        Bogoliuobov de Genne Hamiltonian.

        Parameters
        ----------
        k : :class:`numpy.ndarray`
            List of k points.

        Returns
        -------
        :class:`numpy.ndarray`
            BdG Hamiltonian.

        """
        if np.isnan(k).any() or np.isinf(k).any():
            msg = "k is NaN or Infinity"
            raise ValueError(msg)
        if k.ndim == 1:
            h = self._bdg_hamiltonian_one_point(k)
        else:
            h = np.array([self._bdg_hamiltonian_one_point(k) for k in k])
        return h

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
        if k.ndim == 1:
            h = self._hamiltonian_one_point(k)
        else:
            h = np.array([self._hamiltonian_one_point(k) for k in k])
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
        if k.ndim == 1:
            h = self._hamiltonian_derivative_one_point(k, direction)
        else:
            h = np.array([self._hamiltonian_derivative_one_point(k, direction) for k in k])
        return h

    def diagonalize_nonint(
        self, k: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Diagonalize the normal state Hamiltonian.

        Parameters
        ----------
        k : :class:`numpy.ndarray`
            List of k points.

        Returns
        -------
        :class:`numpy.ndarray`
            Eigenvalues of the normal state Hamiltonian.
        :class:`numpy.ndarray`
            Diagonalising matrix of the normal state Hamiltonian.

        """
        k_point_matrix = self.hamiltonian(k)

        if k.ndim == 1:
            band_energies, bloch_wavefunctions = np.linalg.eigh(k_point_matrix)
        else:
            bloch_wavefunctions = np.zeros(
                (len(k), self.number_of_bands, self.number_of_bands),
                dtype=complex,
            )
            band_energies = np.zeros((len(k), self.number_of_bands))

            for i in range(len(k)):
                band_energies[i], bloch_wavefunctions[i] = np.linalg.eigh(k_point_matrix[i])

        return band_energies, bloch_wavefunctions

    def diagonalize_bdg(
        self, k: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex64]]:
        """
        Diagonalize the BdG Hamiltonian.

        Parameters
        ----------
        k : :class:`numpy.ndarray`
            List of k points.

        Returns
        -------
        :class:`numpy.ndarray`
            Eigenvalues of the BdG Hamiltonian.
        :class:`numpy.ndarray`
            Diagonalising matrix of the BdG Hamiltonian.

        """
        bdg_matrix = self.bdg_hamiltonian(k)

        if k.ndim == 1:
            bdg_energies, bdg_wavefunctions = np.linalg.eigh(bdg_matrix)
        else:
            bdg_wavefunctions = np.zeros(
                (len(k), 2 * self.number_of_bands, 2 * self.number_of_bands),
                dtype=np.complex64,
            )
            bdg_energies = np.zeros((len(k), 2 * self.number_of_bands))

            for i in range(len(k)):
                bdg_energies[i], bdg_wavefunctions[i] = np.linalg.eigh(bdg_matrix[i])

        return bdg_energies, bdg_wavefunctions

    def calculate_bandstructure(
        self,
        k: npt.NDArray[np.float64],
        overlaps: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the band structure.

        Parameters
        ----------
        k : :class:`numpy.ndarray`
            List of k points.
        overlaps : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`), optional
            Overlaps.

        Returns
        -------
        `pandas.DataFrame`
            Band structure.

        """
        k_point_matrix = self.hamiltonian(k)

        results = pd.DataFrame(
            index=range(len(k)),
            dtype=float,
        )

        for i in range(len(k)):
            energies, eigenvectors = np.linalg.eigh(k_point_matrix[i])

            for band_index in range(self.number_of_bands):
                results.loc[i, f"band_{band_index}"] = energies[band_index]

                if overlaps is not None:
                    results.loc[i, f"wx_{band_index}"] = (
                        np.abs(np.dot(eigenvectors[:, band_index], overlaps[0])) ** 2
                        - np.abs(np.dot(eigenvectors[:, band_index], overlaps[1])) ** 2
                    )

        return results
