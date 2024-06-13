import pathlib
from abc import ABC, abstractmethod
from typing import Any, Tuple

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd


class BaseHamiltonian(ABC):
    """Base class for Hamiltonians."""

    def __init__(self, *args: Tuple[Any], **kwargs: dict[str, Any]) -> None:
        pass

    @property
    @abstractmethod
    def number_of_bands(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def coloumb_orbital_basis(self) -> list[float]:
        raise NotImplementedError

    @property
    @abstractmethod
    def delta_orbital_basis(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @delta_orbital_basis.setter
    @abstractmethod
    def delta_orbital_basis(self, new_delta: npt.NDArray[np.float64]) -> None:
        raise NotImplementedError

    @abstractmethod
    def _hamiltonian_k_space_one_point(
        self, k_point: npt.NDArray[np.float64], matrix_in: npt.NDArray[np.complex64]
    ) -> npt.NDArray[np.complex64]:
        """Calculates

        This method is system-specific, so it needs to be implemented in every subclass

        Args:
            k_point:
            matrix_in:

        Returns:

        """
        raise NotImplementedError

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

    def diagonalize_bdg(
        self, k_list: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex64]]:
        bdg_matrix = np.array(
            [
                np.block(
                    [
                        [
                            self.hamiltonian_k_space(k)[0],
                            delta * np.eye(self.number_of_bands),
                        ],
                        [
                            np.conjugate(delta * np.eye(self.number_of_bands)),
                            -np.conjugate(self.hamiltonian_k_space(k)[0]),
                        ],
                    ]
                )
                for k in k_list
            ]
        )
        eigenvalues, eigenvectors = np.linalg.eigh(bdg_matrix)

        return eigenvalues, eigenvectors

    def hamiltonian_k_space(
        self, k: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.complex64]:
        if np.isnan(k).any() or np.isinf(k).any():
            raise ValueError("k is NaN or Infinity")
        if k.ndim == 1:
            h = np.zeros((1, self.number_of_bands, self.number_of_bands), dtype=complex)
            h[0] = self._hamiltonian_k_space_one_point(k, h[0])
        else:
            h = np.zeros(
                (k.shape[0], self.number_of_bands, self.number_of_bands), dtype=complex
            )
            for k_index, k in enumerate(k):
                h[k_index] = self._hamiltonian_k_space_one_point(k, h[k_index])
        return h

    def calculate_bandstructure(
        self,
        k_point_list: npt.NDArray[np.float64],
        overlaps: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    ) -> pd.DataFrame:
        k_point_matrix = self.hamiltonian_k_space(k_point_list)

        results = pd.DataFrame(
            index=range(len(k_point_list)),
            dtype=float,
        )

        for i, k in enumerate(k_point_list):
            energies, eigenvectors = np.linalg.eigh(k_point_matrix[i])

            for band_index in range(self.number_of_bands):
                results.at[i, f"band_{band_index}"] = energies[band_index]

                if overlaps is not None:
                    results.at[i, f"wx_{band_index}"] = (
                        np.abs(np.dot(eigenvectors[:, band_index], overlaps[0])) ** 2
                        - np.abs(np.dot(eigenvectors[:, band_index], overlaps[1])) ** 2
                    )

        return results

    def generate_bloch(
        self, k_points: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        k_point_matrix = self.hamiltonian_k_space(k_points)

        if k_points.ndim == 1:
            energies, bloch = np.linalg.eigh(k_point_matrix[0])
        else:
            bloch = np.zeros(
                (len(k_points), self.number_of_bands, self.number_of_bands),
                dtype=complex,
            )
            energies = np.zeros((len(k_points), self.number_of_bands))

            for i, k in enumerate(k_points):
                energies[i], bloch[i] = np.linalg.eigh(k_point_matrix[i])

        return energies, bloch
