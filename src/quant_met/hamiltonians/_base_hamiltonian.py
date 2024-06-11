from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy._typing import _64Bit
from scipy import optimize


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
        self, k_point_list: npt.NDArray[np.float64]
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

    @staticmethod
    def free_energy(
        delta_vector: npt.NDArray[np.float64],
        diagonalize_bdg: Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ],
        nonint_hamiltonian_k_space: npt.NDArray[np.float64],
        coloumb_orbital_basis: npt.NDArray[np.float64],
        k_points: npt.NDArray[np.float64],
        beta: float | None = None,
    ) -> float:
        number_k_points = len(k_points)
        bdg_energies, bdg_vectors = diagonalize_bdg(k_points, delta_vector)

        k_array: npt.NDArray[np.float64] = np.real(
            np.trace(nonint_hamiltonian_k_space, axis1=-2, axis2=-1)
        ) + np.ones(number_k_points) * np.sum(
            np.power(np.abs(delta_vector), 2) / coloumb_orbital_basis
        )
        if beta is None:
            k_array -= 0.5 * np.array(
                [
                    np.real(
                        np.trace(
                            bdg_vectors[k_index]
                            @ np.diagflat(np.abs(bdg_energies[k_index]))
                            @ np.conjugate(bdg_vectors[k_index]).T
                        )
                    )
                    for k_index in range(number_k_points)
                ]
            )
        else:
            k_array -= (
                np.sum(np.log(1 + np.nan_to_num(np.exp(-beta * bdg_energies))), axis=-1)
                / beta
            )
        integral: float = np.sum(k_array, axis=-1) / (
            2.5980762113533156 * number_k_points
        )

        return integral

    def minimize_loop(
        self, k_points: npt.NDArray[np.float64], beta: float | None = None
    ) -> npt.NDArray[np.float64]:
        nonint_hamiltonian_k_space = self.hamiltonian_k_space(k_points)
        solution = optimize.brute(
            func=self.free_energy,
            args=(
                self.diagonalize_bdg,
                nonint_hamiltonian_k_space,
                self.coloumb_orbital_basis,
                k_points,
                beta,
            ),
            ranges=[(0, 1) for _ in range(self.number_of_bands)],
            Ns=20,
            workers=10,
            finish=optimize.fmin,
            full_output=True,
        )

        delta_solution: npt.NDArray[np.float64] = solution[0]

        return delta_solution
