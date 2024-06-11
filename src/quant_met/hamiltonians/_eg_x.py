import numpy as np
import numpy.typing as npt
import pandas as pd

from ._base_hamiltonian import BaseHamiltonian
from ._utils import _check_valid_float


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
        self.t_gr = _check_valid_float(t_gr, "Hopping graphene")
        self.t_x = _check_valid_float(t_x, "Hopping impurity")
        self.V = _check_valid_float(V, "Hybridisation")
        self.a = _check_valid_float(a, "Lattice constant")
        self.mu = _check_valid_float(mu, "Chemical potential")
        self.U_gr = _check_valid_float(U_gr, "Coloumb interaction graphene")
        self.U_x = _check_valid_float(U_x, "Coloumb interaction impurity")

    @property
    def coloumb_orbital_basis(self) -> list[float]:
        return [self.U_gr, self.U_gr, self.U_x]

    @property
    def number_of_bands(self) -> int:
        return 3

    def _hamiltonian_k_space_one_point(
        self, k: npt.NDArray[np.float64], h: npt.NDArray[np.complex64]
    ) -> npt.NDArray[np.complex64]:
        t_gr = self.t_gr
        t_x = self.t_x
        a = self.a
        # a_0 = a / np.sqrt(3)
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

    def calculate_bandstructure(
        self, k_point_list: npt.NDArray[np.float64]
    ) -> pd.DataFrame:
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
