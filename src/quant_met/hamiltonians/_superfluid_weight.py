import numpy as np
import numpy.typing as npt

from ._base_hamiltonian import BaseHamiltonian


def calculate_current_operator(
    h: BaseHamiltonian, direction: str, k: npt.NDArray[np.float64]
) -> npt.NDArray[np.complex64]:
    j = np.zeros(shape=(3, 3), dtype=np.complex64)

    _, bloch = h.diagonalize_nonint(k=k)
    bloch = bloch[0]

    for m in range(3):
        for n in range(3):
            j[m, n] = (
                np.conjugate(bloch[:, m])
                @ h.hamiltonian_derivative(direction=direction, k=k)
                @ bloch[:, n]
            )

    return j


def calculate_w_matrix(
    h: BaseHamiltonian, k: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.complex64], npt.NDArray[np.complex64]]:
    bloch = h.diagonalize_nonint(k=k)
    _, bdg_functions = h.diagonalize_bdg(k=k)
    bdg_functions = bdg_functions[0]

    w_plus = np.zeros((4, 2), dtype=np.complex64)
    for i in range(0, 4):
        for m in range(0, 2):
            w_plus[i, m] = (
                np.tensordot(bloch[m], np.array([1, 0]), axes=0).reshape(-1)
                @ bdg_functions[i]
            )

    w_minus = np.zeros((4, 2), dtype=np.complex64)
    for i in range(0, 4):
        for m in range(0, 2):
            w_minus[i, m] = (
                np.tensordot(np.conjugate(bloch[m]), np.array([0, 1]), axes=0).reshape(
                    -1
                )
                @ bdg_functions[i]
            )

    return w_plus, w_minus


def calculate_c_factor(
    h: BaseHamiltonian, k: npt.NDArray[np.float64]
) -> npt.NDArray[np.complex64]:
    bdg_energies, _ = h.diagonalize_bdg(k)
    w_plus, w_minus = calculate_w_matrix(h, k)
    bdg_energies = bdg_energies[0]
    C_mnpq = np.zeros(shape=(3, 3, 3, 3), dtype=np.complex64)

    for m in range(3):
        for n in range(3):
            for p in range(3):
                for q in range(3):
                    C_tmp: float = 0
                    for i in range(6):
                        for j in range(6):
                            if bdg_energies[i] != bdg_energies[j]:
                                C_tmp += (
                                    fermi_dirac(bdg_energies[i])
                                    - fermi_dirac(bdg_energies[j])
                                ) / (bdg_energies[j] - bdg_energies[i])
                            else:
                                C_tmp -= fermi_dirac_derivative(bdg_energies[i])

                            C_tmp *= (
                                np.conjugate(w_minus[i, m])
                                * w_plus[j, n]
                                * np.conjugate(w_minus[j, p])
                                * w_minus[i, q]
                            )

                    C_mnpq[m, n, p, q] = 2 * C_tmp

    return C_mnpq


def fermi_dirac_derivative(energy: np.float64, beta: np.float64 | None = None) -> float:
    if beta is None:
        if energy == 0:
            return np.inf
        else:
            return 0
    else:
        raise NotImplementedError


def fermi_dirac(energy: np.float64, beta: np.float64 | None = None) -> np.float64:
    if beta is None:
        return_value: np.float64 = np.heaviside(energy, 0)
    else:
        raise NotImplementedError

    return return_value


def calculate_superfluid_weight(
    h: BaseHamiltonian,
    k_grid: npt.NDArray[np.float64],
    direction_1: str,
    direction_2: str,
) -> tuple[float, float]:
    number_k_points = len(k_grid)

    s_weight_conv = 0
    s_weight_geom = 0

    for k in k_grid:
        C_mnpq = calculate_c_factor(h, k)
        j_up = calculate_current_operator(h, direction_1, k)
        j_down = calculate_current_operator(h, direction_2, -k)
        for m in range(3):
            for n in range(3):
                for p in range(3):
                    for q in range(3):
                        s_weight = C_mnpq[m, n, p, q] * j_up[m, n] * j_down[q, p]
                        if m == n and p == q:
                            s_weight_conv += s_weight
                        else:
                            s_weight_geom += s_weight

    return s_weight_conv / number_k_points, s_weight_geom / number_k_points
