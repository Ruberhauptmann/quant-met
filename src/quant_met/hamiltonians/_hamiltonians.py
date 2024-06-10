from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import pandas as pd
import sympy as sp


def _check_valid_float(float_in: float, parameter_name: str) -> float:
    if np.isinf(float_in):
        raise ValueError(f"{parameter_name} must not be Infinity")
    elif np.isnan(float_in):
        raise ValueError(f"{parameter_name} must not be NaN")
    else:
        return float_in


class BaseHamiltonian(ABC):
    @property
    @abstractmethod
    def nonint_hamiltonian(self):
        raise NotImplementedError

    @abstractmethod
    def eval_nonint_hamiltonian(self, *args, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self):
        raise NotImplementedError

    def bdg_hamiltonian(self, k, delta):
        bdg_matrix = np.block(
            [
                [
                    self.eval_nonint_hamiltonian(k)[0],
                    delta * np.eye(self.number_of_bands),
                ],
                [
                    np.conjugate(delta * np.eye(self.number_of_bands)),
                    -np.conjugate(self.eval_nonint_hamiltonian(k)[0]),
                ],
            ]
        )
        return bdg_matrix

    def diagonalize_bdg(
        self, k_list: npt.NDArray[np.float64], delta: npt.NDArray[np.float64], **kwargs
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex64]]:
        bdg_matrix = np.array([self.bdg_hamiltonian(k, delta) for k in k_list])
        eigenvalues, eigenvectors = np.linalg.eigh(bdg_matrix)

        return eigenvalues, eigenvectors

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

    def calculate_bloch_functions(
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


class GrapheneHamiltonian(BaseHamiltonian):
    parameters = {
        "t_nn": (
            sp.symbols("t_nn", real=True, positive=True),
            "Hopping between nearest neighbours",
        ),
        "mu": (sp.symbols("mu", real=True), "Chemical potential"),
        "a": (sp.symbols("a", real=True, positive=True), "Lattice constant"),
        "U": (sp.symbols("U", real=True), "Local coloumb interaction"),
        "k": (
            sp.Matrix([sp.symbols("k_x, k_y", real=True)]),
            "Lattice momentum vector",
        ),
    }

    def eval_nonint_hamiltonian(
        self, k: npt.NDArray[np.float64], t_nn, mu, a
    ) -> npt.NDArray[np.complex64]:
        t_nn = _check_valid_float(t_nn, "Hopping")
        if a <= 0:
            raise ValueError("Lattice constant must be positive")
        a = _check_valid_float(a, "Lattice constant")
        mu = _check_valid_float(mu, "Chemical potential")
        if np.isnan(k).any() or np.isinf(k).any():
            raise ValueError("k is NaN or Infinity")
        h_eval = sp.lambdify(
            args=[
                self.parameters["t_nn"],
                self.parameters["mu"],
                self.parameters["a"],
                self.parameters["k"][0],
                self.parameters["k"][1],
            ],
            expr=sp.ImmutableDenseMatrix(self.nonint_hamiltonian),
            modules="numpy",
        )
        h = np.transpose(
            h_eval(
                t_nn=t_nn * np.ones(k.shape[0]),
                mu=mu * np.ones(k.shape[0]),
                a=a * np.ones(k.shape[0]),
                k_x=k.T[0],
                k_y=k.T[1],
            ),
            axes=(2, 0, 1),
        )
        return h

    @property
    def nonint_hamiltonian(self):
        t_nn = self.parameters["t_nn"][0]
        mu = self.parameters["mu"][0]
        a = self.parameters["a"][0]
        k_x = self.parameters["k"][0][0]
        k_y = self.parameters["k"][0][1]

        f_gr = t_nn * (
            sp.exp(1j * k_y * a / sp.sqrt(3))
            + 2 * sp.exp(-0.5j * a / sp.sqrt(3) * k_y) * (sp.cos(0.5 * a * k_x))
        )
        hamiltonian = sp.Matrix([[-mu, f_gr], [sp.conjugate(f_gr), -mu]])
        return hamiltonian

    def eval_nonint_hamiltonian_numpy(
        self, k: npt.NDArray[np.float64], h: npt.NDArray[np.complex64]
    ) -> npt.NDArray[np.complex64]:
        """
        t_nn = self.t_nn
        a = self.a
        mu = self.mu
        """
        t_nn = 1
        a = np.sqrt(3)
        mu = 1

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
