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
    def hubbard_int_orbital_basis(self) -> npt.NDArray[np.float64]:
        """
        hubbard_int interaction split up in orbitals.

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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    def save(self, filename: pathlib.Path) -> None:
        """
        Save the Hamiltonian as a HDF5 file.

        Parameters
        ----------
        filename : :class:`pathlib.Path`
            Filename to save the Hamiltonian to, should end in .hdf5

        """
        with h5py.File(f"{filename}", "w") as f:
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
        assert _check_valid_array(k)
        if k.ndim == 1:
            k = np.expand_dims(k, axis=0)

        h = np.zeros(
            (k.shape[0], 2 * self.number_of_bands, 2 * self.number_of_bands), dtype=np.complex64
        )

        h[:, 0 : self.number_of_bands, 0 : self.number_of_bands] = self.hamiltonian(k)
        h[
            :,
            self.number_of_bands : 2 * self.number_of_bands,
            self.number_of_bands : 2 * self.number_of_bands,
        ] = -self.hamiltonian(-k).conjugate()

        for i in range(self.number_of_bands):
            h[:, self.number_of_bands + i, i] = self.delta_orbital_basis[i]
        h[:, 0 : self.number_of_bands, self.number_of_bands : self.number_of_bands * 2] = (
            h[:, self.number_of_bands : self.number_of_bands * 2, 0 : self.number_of_bands]
            .copy()
            .conjugate()
        )

        return h.squeeze()

    def bdg_hamiltonian_derivative(
        self, k: npt.NDArray[np.float64], direction: str
    ) -> npt.NDArray[np.complex64]:
        """
        Deriative of the BdG Hamiltonian.

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
            k = np.expand_dims(k, axis=0)

        h = np.zeros(
            (k.shape[0], 2 * self.number_of_bands, 2 * self.number_of_bands), dtype=np.complex64
        )

        h[:, 0 : self.number_of_bands, 0 : self.number_of_bands] = self.hamiltonian_derivative(
            k, direction
        )
        h[
            :,
            self.number_of_bands : 2 * self.number_of_bands,
            self.number_of_bands : 2 * self.number_of_bands,
        ] = -self.hamiltonian_derivative(-k, direction).conjugate()

        return h.squeeze()

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
        if k_point_matrix.ndim == 2:
            k_point_matrix = np.expand_dims(k_point_matrix, axis=0)
            k = np.expand_dims(k, axis=0)

        bloch_wavefunctions = np.zeros(
            (len(k), self.number_of_bands, self.number_of_bands),
            dtype=complex,
        )
        band_energies = np.zeros((len(k), self.number_of_bands))

        for i in range(len(k)):
            band_energies[i], bloch_wavefunctions[i] = np.linalg.eigh(k_point_matrix[i])

        return band_energies.squeeze(), bloch_wavefunctions.squeeze()

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
        if bdg_matrix.ndim == 2:
            bdg_matrix = np.expand_dims(bdg_matrix, axis=0)
            k = np.expand_dims(k, axis=0)

        bdg_wavefunctions = np.zeros(
            (len(k), 2 * self.number_of_bands, 2 * self.number_of_bands),
            dtype=np.complex64,
        )
        bdg_energies = np.zeros((len(k), 2 * self.number_of_bands))

        for i in range(len(k)):
            bdg_energies[i], bdg_wavefunctions[i] = np.linalg.eigh(bdg_matrix[i])

        return bdg_energies.squeeze(), bdg_wavefunctions.squeeze()

    def gap_equation(
        self, k: npt.NDArray[np.float64], beta: np.float64
    ) -> npt.NDArray[np.complex64]:
        """Gap equation.

        Parameters
        ----------
        beta
        k

        Returns
        -------
        :class:`numpy.ndarray`
            New gap in orbital basis.


        """
        bdg_energies, bdg_wavefunctions = self.diagonalize_bdg(k)
        bdg_energies_minus_k, _ = self.diagonalize_bdg(-k)
        delta = np.zeros(self.number_of_bands, dtype=np.complex64)

        for i in range(self.number_of_bands):
            sum_tmp = 0
            for j in range(self.number_of_bands):
                for k_index in range(len(k)):
                    sum_tmp += np.conjugate(bdg_wavefunctions[k_index, i, j]) * bdg_wavefunctions[
                        k_index, i + self.number_of_bands, j
                    ] * _fermi_dirac(
                        bdg_energies[k_index, j + self.number_of_bands].item(), beta
                    ) + np.conjugate(
                        bdg_wavefunctions[k_index, i, j + self.number_of_bands]
                    ) * bdg_wavefunctions[
                        k_index, i + self.number_of_bands, j + self.number_of_bands
                    ] * _fermi_dirac(
                        -bdg_energies_minus_k[k_index, j + self.number_of_bands].item(), beta
                    )
            delta[i] = (-self.hubbard_int_orbital_basis[i] * sum_tmp / len(k)).conjugate()

        return np.abs(delta)

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
        results = pd.DataFrame(
            index=range(len(k)),
            dtype=float,
        )
        energies, wavefunctions = self.diagonalize_nonint(k)

        for i, (energy_k, wavefunction_k) in enumerate(zip(energies, wavefunctions, strict=False)):
            if self.number_of_bands == 1:
                results.loc[i, "band"] = energy_k
            else:
                for band_index in range(self.number_of_bands):
                    results.loc[i, f"band_{band_index}"] = energy_k[band_index]

                    if overlaps is not None:
                        results.loc[i, f"wx_{band_index}"] = (
                            np.abs(np.dot(wavefunction_k[:, band_index], overlaps[0])) ** 2
                            - np.abs(np.dot(wavefunction_k[:, band_index], overlaps[1])) ** 2
                        )

        return results


def _fermi_dirac(energy: np.float64, beta: np.float64) -> np.float64:
    fermi_dirac: np.float64 = 1 / (1 + np.exp(beta * energy))
    return fermi_dirac
