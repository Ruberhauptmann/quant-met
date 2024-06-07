from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import pandas as pd


def check_valid_float(float_in: float, parameter_name: str) -> float:
    if np.isinf(float_in):
        raise ValueError(f"{parameter_name} must not be Infinity")
    elif np.isnan(float_in):
        raise ValueError(f"{parameter_name} must not be NaN")
    else:
        return float_in


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

    def diagonalize_bdg(self, k_list: npt.NDArray, delta: npt.NDArray):
        bdg_matrix = np.array(
            [
                np.block(
                    [
                        [
                            self.hamiltonian_k_space(k)[0],
                            delta * np.eye(self.number_of_bands),
                        ],
                        [
                            delta * np.eye(self.number_of_bands),
                            np.conjugate(self.hamiltonian_k_space(k)[0]),
                        ],
                    ]
                )
                for k in k_list
            ]
        )
        eigenvalues, eigenvectors = np.linalg.eigh(bdg_matrix)

        return eigenvalues, eigenvectors

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
        k_point_matrix = self.hamiltonian_k_space(k_point_list)

        results = pd.DataFrame(
            index=range(len(k_point_list)),
            dtype=float,
        )

        for i, k in enumerate(k_point_list):
            energies, eigenvectors = np.linalg.eigh(k_point_matrix[i])

            for band_index in range(self.number_of_bands):
                results.at[i, f"band_{band_index}"] = energies[band_index]

        return results

    def generate_bloch(self, k_points: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
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


class GrapheneHamiltonian(BaseHamiltonian):
    def __init__(self, t_nn: float, a: float, mu: float, coulomb_gr: float):
        self.t_nn = check_valid_float(t_nn, "Hopping")
        if a <= 0:
            raise ValueError("Lattice constant must be positive")
        self.a = check_valid_float(a, "Lattice constant")
        self.mu = check_valid_float(mu, "Chemical potential")
        self.coloumb_gr = check_valid_float(coulomb_gr, "Coloumb interaction")

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
        self.t_gr = check_valid_float(t_gr, "Hopping graphene")
        self.t_x = check_valid_float(t_x, "Hopping impurity")
        self.V = check_valid_float(V, "Hybridisation")
        self.a = check_valid_float(a, "Lattice constant")
        self.mu = check_valid_float(mu, "Chemical potential")
        self.U_gr = check_valid_float(U_gr, "Coloumb interaction graphene")
        self.U_x = check_valid_float(U_x, "Coloumb interaction impurity")

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

        return np.nan_to_num(h)

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
