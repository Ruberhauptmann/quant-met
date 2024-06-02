"""

"""
from abc import ABC, abstractmethod

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
    def coloumb_orbital_basis(self) -> list[float]:
        """

        Returns:
            list[float]:
        """
        raise NotImplementedError

    @abstractmethod
    def _hamiltonian_k_space_one_point(
        self, k_point: npt.NDArray, matrix_in: npt.NDArray
    ) -> npt.NDArray:
        """Calculates

        This method is system-specific, so it needs to be implemented in every subclass

        Args:
            k_point:
            matrix_in:

        Returns:

        """
        raise NotImplementedError

    def hamiltonian_k_space(self, k: npt.NDArray) -> npt.NDArray:
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

    def calculate_bandstructure(self, k_point_list: npt.NDArray) -> pd.DataFrame:
        k_point_matrix = self.k_space_matrix(k_point_list)

        results = pd.DataFrame(
            index=range(len(k_point_list)),
            dtype=float,
        )

        for i, k in enumerate(k_point_list):
            energies, eigenvectors = np.linalg.eigh(k_point_matrix[i])

            for band_index in range(self.number_bands):
                results.at[i, f"band_{band_index}"] = energies[band_index]

        return results

    def generate_bloch(self, k_points: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        k_point_matrix = self.k_space_matrix(k_points)

        if k_points.ndim == 1:
            energies, bloch = np.linalg.eigh(k_point_matrix[0])
        else:
            bloch = np.zeros(
                (len(k_points), self.number_bands, self.number_bands), dtype=complex
            )
            energies = np.zeros((len(k_points), self.number_bands))

            for i, k in enumerate(k_points):
                energies[i], bloch[i] = np.linalg.eigh(k_point_matrix[i])

        return energies, bloch


class GrapheneHamiltonian(BaseHamiltonian):
    def __init__(self, t_nn: float, a: float, mu: float, coulomb_gr: float):
        if np.isinf(t_nn) or np.isnan(t_nn):
            raise ValueError("Hopping must not be NaN or Infinity")
        self.t_nn = t_nn
        if a <= 0:
            raise ValueError("Lattice constant must be positive")
        if np.isinf(a) or np.isnan(a):
            raise ValueError("Lattice constant must not be NaN or Infinity")
        self.a = a
        if np.isinf(mu) or np.isnan(mu):
            raise ValueError("Chemical potential must not be NaN or Infinity")
        self.mu = mu
        if np.isinf(coulomb_gr) or np.isnan(coulomb_gr):
            raise ValueError("Coloumb interaction must not be NaN or Infinity")
        self.coloumb_gr = coulomb_gr

    @property
    def coloumb_orbital_basis(self) -> list[float]:
        return [self.coloumb_gr, self.coloumb_gr]

    @property
    def number_of_bands(self) -> int:
        return 2

    def _hamiltonian_k_space_one_point(
        self, k: npt.NDArray, h: npt.NDArray
    ) -> npt.NDArray:
        t_nn = self.t_nn
        a = self.a
        mu = self.mu

        h[0, 1] = t_nn * (
            np.exp(1j * k[1] * a / np.sqrt(3))
            + 2 * np.exp(-0.5j * a / np.sqrt(3) * k[1]) * (np.cos(0.5 * a * k[0]))
        )

        h[1, 0] = h[0, 1].conjugate()
        h = h - mu * np.eye(2)

        return np.nan_to_num(h)


class EGXHamiltonian(BaseHamiltonian):
    def __init__(
        self,
        t_gr: float,
        t_x: float,
        V: float,
        a: float,
        mu: float,
        U_gr: float,
        U_x: float,
    ):
        self.t_gr = t_gr
        self.t_x = t_x
        self.V = V
        self.a = a
        self.mu = mu
        self.U_gr = U_gr
        self.U_x = U_x

    @property
    def coloumb_orbital_basis(self) -> list[float]:
        return [self.U_gr, self.U_gr, self.U_x]

    @property
    def number_of_bands(self) -> int:
        return 3

    def _hamiltonian_k_space_one_point(
        self, k: npt.NDArray, h: npt.NDArray
    ) -> npt.NDArray:
        t_gr = self.t_gr
        t_x = self.t_x
        a = self.a
        a_0 = a / np.sqrt(3)
        V = self.V
        mu = self.mu

        h[0, 1] = t_gr * (
            np.exp(1j * k[1] * a / np.sqrt(3))
            + 2 * np.exp(-0.5j * a / np.sqrt(3) * k[1]) * (np.cos(0.5 * a * k[0]))
        )

        h[1, 0] = h[0, 1].conjugate()

        h[2, 0] = V
        h[0, 2] = V

        h[2, 2] = (
            -2
            * t_x
            * (
                np.cos(a * k[0])
                + 2 * np.cos(0.5 * a * k[0]) * np.cos(0.5 * np.sqrt(3) * a * k[1])
            )
        )
        h = h - mu * np.eye(3)
        return h

    def calculate_bandstructure(self, k_point_list: npt.NDArray) -> pd.DataFrame:
        """

        Args:
             k_point_list (npt.NDArray): Test
        """
        k_point_matrix = self.hamiltonian_k_space(k_point_list)

        results = pd.DataFrame(
            index=range(len(k_point_list)),
            dtype=float,
        )

        for i, k in enumerate(k_point_list):
            energies, eigenvectors = np.linalg.eigh(k_point_matrix[i])

            for band_index in range(self.number_of_bands):
                results.at[i, f"band_{band_index}"] = energies[band_index]
                results.at[i, f"wx_{band_index}"] = (
                    np.abs(np.dot(eigenvectors[:, band_index], np.array([0, 0, 1])))
                    ** 2
                    - np.abs(np.dot(eigenvectors[:, band_index], np.array([1, 0, 0])))
                    ** 2
                )

        return results
