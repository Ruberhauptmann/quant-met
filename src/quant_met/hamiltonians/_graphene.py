import numpy as np
import numpy.typing as npt

from ._base_hamiltonian import BaseHamiltonian
from ._utils import _check_valid_float


class GrapheneHamiltonian(BaseHamiltonian):
    def __init__(
        self,
        t_nn: float,
        a: float,
        mu: float,
        coulomb_gr: float,
        delta: npt.NDArray[np.float64] | None = None,
    ):
        self.t_nn = _check_valid_float(t_nn, "Hopping")
        if a <= 0:
            raise ValueError("Lattice constant must be positive")
        self.a = _check_valid_float(a, "Lattice constant")
        self.mu = _check_valid_float(mu, "Chemical potential")
        self.coulomb_gr = _check_valid_float(coulomb_gr, "Coloumb interaction")
        if delta is None:
            self._delta_orbital_basis = np.zeros(2)
        else:
            self._delta_orbital_basis = delta

    @property
    def coloumb_orbital_basis(self) -> npt.NDArray[np.float64]:
        return np.array([self.coulomb_gr, self.coulomb_gr])

    @property
    def number_of_bands(self) -> int:
        return 2

    @property
    def delta_orbital_basis(self) -> npt.NDArray[np.float64]:
        return self._delta_orbital_basis

    @delta_orbital_basis.setter
    def delta_orbital_basis(self, new_delta: npt.NDArray[np.float64]) -> None:
        self._delta_orbital_basis = new_delta

    def _hamiltonian_derivative_one_point(
        self, k: npt.NDArray[np.float64], direction: str
    ) -> npt.NDArray[np.complex64]:
        assert direction in ["x", "y"]

        t_nn = self.t_nn
        a = self.a

        h = np.zeros((self.number_of_bands, self.number_of_bands), dtype=np.complex64)

        if direction == "x":
            h[0, 1] = (
                t_nn
                * a
                * np.exp(-0.5j * a / np.sqrt(3) * k[1])
                * np.sin(0.5 * a * k[0])
            )
            h[1, 0] = h[0, 1].conjugate()
        elif direction == "y":
            h[0, 1] = (
                -t_nn
                * 1j
                * a
                / np.sqrt(3)
                * (
                    np.exp(1j * a / np.sqrt(3) * k[1])
                    - np.exp(-0.5j * a / np.sqrt(3) * k[1]) * np.cos(0.5 * a * k[0])
                )
            )
            h[1, 0] = h[0, 1].conjugate()

        return h

    def _hamiltonian_one_point(
        self, k: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.complex64]:
        t_nn = self.t_nn
        a = self.a
        mu = self.mu

        h = np.zeros((self.number_of_bands, self.number_of_bands), dtype=np.complex64)

        h[0, 1] = -t_nn * (
            np.exp(1j * k[1] * a / np.sqrt(3))
            + 2 * np.exp(-0.5j * a / np.sqrt(3) * k[1]) * (np.cos(0.5 * a * k[0]))
        )
        h[1, 0] = h[0, 1].conjugate()
        h -= mu * np.eye(2)

        return h
