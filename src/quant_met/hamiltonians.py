from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import pandas as pd

from quant_met.configuration import Configuration


class BaseHamiltonian(ABC):
    @property
    @abstractmethod
    def number_bands(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def k_space_matrix(self, k: npt.NDArray, *args, **kwargs) -> npt.NDArray:
        raise NotImplementedError

    def calculate_bandstructure(self, k_point_list: npt.NDArray):
        k_point_matrix = self.k_space_matrix(k_point_list)

        number_bands = self.k_space_matrix(k_point_list).shape[-1]

        results = pd.DataFrame(
            index=range(len(k_point_list)),
            dtype=float,
        )

        for i, k in enumerate(k_point_list):
            energies, eigenvectors = np.linalg.eigh(k_point_matrix[i])

            for band_index in range(number_bands):
                results.at[i, f"band_{band_index}"] = energies[band_index]

        return results

    def generate_bloch(self, k_points: npt.NDArray, mu: float):
        k_point_matrix = self.k_space_matrix(k_points)

        bloch = np.zeros((len(k_points), 2, 2), dtype=complex)
        energies = np.zeros((len(k_points), 2))

        for i, k in enumerate(k_points):
            energies[i], bloch[i] = np.linalg.eigh(k_point_matrix[i])
            energies[i] = energies[i] - mu

        bloch_absolute = np.power(np.absolute(bloch), 2)

        return energies, bloch_absolute


class GrapheneHamiltonian(BaseHamiltonian):
    def __init__(self, t_nn: float, a: float, mu: float):
        self.t_nn = t_nn
        self.a = a
        self.mu = mu

    @property
    def number_bands(self) -> int:
        return 2

    def k_space_matrix(self, k: npt.NDArray) -> npt.NDArray:
        t_nn = self.t_nn
        a = self.a

        if k.ndim == 1:
            h = np.zeros((2, 2), dtype=complex)

            h[0, 1] = t_nn * (
                np.exp(1j * k[1] * a / np.sqrt(3))
                + 2 * np.exp(-0.5j * a / np.sqrt(3) * k[1]) * (np.cos(0.5 * a * k[0]))
            )

            h[1, 0] = h[0, 1].conjugate()
        else:
            h = np.zeros((k.shape[0], 2, 2), dtype=complex)

            for k_index, k in enumerate(k):
                h[k_index, 0, 1] = t_nn * (
                    np.exp(1j * k[1] * a / np.sqrt(3))
                    + 2
                    * np.exp(-0.5j * a / np.sqrt(3) * k[1])
                    * (np.cos(0.5 * a * k[0]))
                )

                h[k_index, 1, 0] = h[k_index, 0, 1].conjugate()

        return h


class EGXHamiltonian:
    def generate_non_interacting_hamiltonian(k: npt.NDArray, config: Configuration):
        h = np.zeros((3, 3), dtype=complex)

        t_gr = config.t_gr
        t_x = config.t_x
        a = config.a
        V = config.V

        ## Double counting?
        # h[0, 1] = -0.5 * t_gr * (np.exp(1j * k[1] * a / np.sqrt(3)) + 2 * np.exp(-0.5j * a / np.sqrt(3) * k[1]) * (np.cos(0.5 * a * k[0])))
        h[0, 1] = t_gr * (
            np.exp(1j * k[1] * a / np.sqrt(3))
            + 2 * np.exp(-0.5j * a / np.sqrt(3) * k[1]) * (np.cos(0.5 * a * k[0]))
        )

        h[1, 0] = h[0, 1].conjugate()

        h[2, 1] = V
        h[1, 2] = V

        # h[2, 2] = - t_x * (
        #    np.cos(a * k[0]) +
        #    2 * np.cos(0.5 * a * k[0]) * np.cos(0.5 * np.sqrt(3) * a * k[1])
        # )
        h[2, 2] = (
            -2
            * t_x
            * (
                np.cos(a * k[0])
                + 2 * np.cos(0.5 * a * k[0]) * np.cos(0.5 * np.sqrt(3) * a * k[1])
            )
        )

        h = h - config.mu * np.eye(3)

        return h

    @abstractmethod
    def calculate_bandstructure(self, k_point_list: npt.NDArray):
        k_point_matrix = self.k_space_matrix(k_point_list)

        number_bands = self.k_space_matrix(k_point_list).shape[0]

        results = pd.DataFrame(
            # columns=[ f"band_{i}", f"wx_{i}" for i in range(number_bands)],
            index=range(len(k_point_list)),
            dtype=float,
        )

        for i, k in enumerate(k_point_list):
            energies, eigenvectors = np.linalg.eigh(k_point_matrix[i])

            for band_index in range(number_bands):
                results.at[i, f"band_{band_index}"] = energies[0]
                results.at[i, f"wx_{band_index}"] = (
                    np.abs(np.dot(eigenvectors[:, 0], np.array([0, 0, 1]))) ** 2
                    - np.abs(np.dot(eigenvectors[:, 0], np.array([1, 0, 0]))) ** 2
                )

        return results

    pass
