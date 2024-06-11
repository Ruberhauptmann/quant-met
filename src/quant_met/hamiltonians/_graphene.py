import numpy as np
import numpy.typing as npt

from ._base_hamiltonian import BaseHamiltonian
from ._utils import _check_valid_float


class GrapheneHamiltonian(BaseHamiltonian):
    def __init__(self, t_nn: float, a: float, mu: float, coulomb_gr: float):
        self.t_nn = _check_valid_float(t_nn, "Hopping")
        if a <= 0:
            raise ValueError("Lattice constant must be positive")
        self.a = _check_valid_float(a, "Lattice constant")
        self.mu = _check_valid_float(mu, "Chemical potential")
        self.coloumb_gr = _check_valid_float(coulomb_gr, "Coloumb interaction")

    @property
    def coloumb_orbital_basis(self) -> list[float]:
        return [self.coloumb_gr, self.coloumb_gr]

    @property
    def number_of_bands(self) -> int:
        return 2

    def _hamiltonian_k_space_one_point(
        self, k: npt.NDArray[np.float64], h: npt.NDArray[np.complex64]
    ) -> npt.NDArray[np.complex64]:
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
