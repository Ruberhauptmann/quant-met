import pathlib
from abc import ABC, abstractmethod

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd


class BaseHamiltonian(ABC):
    """Base class for Hamiltonians."""

    @property
    @abstractmethod
    def number_of_bands(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def coloumb_orbital_basis(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @property
    def delta_orbital_basis(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @delta_orbital_basis.setter
    @abstractmethod
    def delta_orbital_basis(self, new_delta: npt.NDArray[np.float64]) -> None:
        raise NotImplementedError

    @abstractmethod
    def _hamiltonian_one_point(
        self, k_point: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.complex64]:
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

        h = np.block(
            [
                [self.hamiltonian(k_point), delta_matrix],
                [np.conjugate(delta_matrix), -np.conjugate(self.hamiltonian(-k_point))],
            ]
        )
        return h

    def save(self, filename: pathlib.Path) -> None:
        with h5py.File(f"{filename}", "a") as f:
            f.create_dataset("delta", data=self.delta_orbital_basis)
            for key, value in vars(self).items():
                if not key.startswith("_"):
                    f.attrs[key] = value

    @classmethod
    def from_file(cls, filename: pathlib.Path) -> "BaseHamiltonian":
        config_dict = {}
        with h5py.File(f"{filename}", "r") as f:
            config_dict["delta"] = f["delta"][()]
            for key, value in f.attrs.items():
                config_dict[key] = value

        return cls(**config_dict)

    def bdg_hamiltonian(self, k: npt.NDArray[np.float64]) -> npt.NDArray[np.complex64]:
        if np.isnan(k).any() or np.isinf(k).any():
            raise ValueError("k is NaN or Infinity")
        if k.ndim == 1:
            h = self._bdg_hamiltonian_one_point(k)
        else:
            h = np.array([self._bdg_hamiltonian_one_point(k) for k in k])
        return h

    def hamiltonian(self, k: npt.NDArray[np.float64]) -> npt.NDArray[np.complex64]:
        if np.isnan(k).any() or np.isinf(k).any():
            raise ValueError("k is NaN or Infinity")
        if k.ndim == 1:
            h = self._hamiltonian_one_point(k)
        else:
            h = np.array([self._hamiltonian_one_point(k) for k in k])
        return h

    def hamiltonian_derivative(
        self, k: npt.NDArray[np.float64], direction: str
    ) -> npt.NDArray[np.complex64]:
        if np.isnan(k).any() or np.isinf(k).any():
            raise ValueError("k is NaN or Infinity")
        if k.ndim == 1:
            h = self._hamiltonian_derivative_one_point(k, direction)
        else:
            h = np.array(
                [self._hamiltonian_derivative_one_point(k, direction) for k in k]
            )
        return h

    def diagonalize_nonint(
        self, k: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        k_point_matrix = self.hamiltonian(k)

        if k.ndim == 1:
            band_energies, bloch_wavefunctions = np.linalg.eigh(k_point_matrix)
        else:
            bloch_wavefunctions = np.zeros(
                (len(k), self.number_of_bands, self.number_of_bands),
                dtype=complex,
            )
            band_energies = np.zeros((len(k), self.number_of_bands))

            for i, k in enumerate(k):
                band_energies[i], bloch_wavefunctions[i] = np.linalg.eigh(
                    k_point_matrix[i]
                )

        return band_energies, bloch_wavefunctions

    def diagonalize_bdg(
        self, k: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex64]]:
        bdg_matrix = self.bdg_hamiltonian(k)

        if k.ndim == 1:
            bdg_energies, bdg_wavefunctions = np.linalg.eigh(bdg_matrix)
        else:
            bdg_wavefunctions = np.zeros(
                (len(k), 2 * self.number_of_bands, 2 * self.number_of_bands),
                dtype=np.complex64,
            )
            bdg_energies = np.zeros((len(k), 2 * self.number_of_bands))

            for i, k in enumerate(k):
                bdg_energies[i], bdg_wavefunctions[i] = np.linalg.eigh(bdg_matrix[i])

        return bdg_energies, bdg_wavefunctions

    def calculate_bandstructure(
        self,
        k: npt.NDArray[np.float64],
        overlaps: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    ) -> pd.DataFrame:
        k_point_matrix = self.hamiltonian(k)

        results = pd.DataFrame(
            index=range(len(k)),
            dtype=float,
        )

        for i, k in enumerate(k):
            energies, eigenvectors = np.linalg.eigh(k_point_matrix[i])

            for band_index in range(self.number_of_bands):
                results.at[i, f"band_{band_index}"] = energies[band_index]

                if overlaps is not None:
                    results.at[i, f"wx_{band_index}"] = (
                        np.abs(np.dot(eigenvectors[:, band_index], overlaps[0])) ** 2
                        - np.abs(np.dot(eigenvectors[:, band_index], overlaps[1])) ** 2
                    )

        return results
