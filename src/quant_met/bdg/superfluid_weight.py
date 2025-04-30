"""Function to calculate superfluid weight."""

import numpy as np
import numpy.typing as npt
import sisl

from .bdg_hamiltonian import bdg_hamiltonian


def calculate_superfluid_weight(
    hamiltonian: sisl.Hamiltonian,
    k_grid: npt.NDArray[np.floating],
    beta: float,
    delta_orbital_basis: npt.NDArray[np.complexfloating],
) -> tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating]]:
    """Calculate superfluid weight (conventional + geometric)."""
    s_weight_conv = np.zeros((2, 2), dtype=np.complex128)
    s_weight_geom = np.zeros((2, 2), dtype=np.complex128)
    c_mnpq_cache = {}

    for i, _dir1 in enumerate(["x", "y"]):
        for j, _dir2 in enumerate(["x", "y"]):
            for k_point in k_grid:
                k_tuple = tuple(k_point)

                # Solve BdG problem
                bdg_h = bdg_hamiltonian(hamiltonian, k_point, delta_orbital_basis)
                energies, wavefuncs = np.linalg.eigh(bdg_h)

                # Cache coefficient tensor
                if k_tuple not in c_mnpq_cache:
                    c_mnpq_cache[k_tuple] = _c_factor(energies, wavefuncs, beta)
                c_mnpq = c_mnpq_cache[k_tuple]

                # Derivatives
                h_deriv_1 = hamiltonian.dHk(k=k_point, format="array")[i]
                h_deriv_2 = hamiltonian.dHk(k=-k_point, format="array")[j]

                j_op_1 = _current_operator(h_deriv_1, wavefuncs)
                j_op_2 = _current_operator(h_deriv_2, wavefuncs)

                for m in range(len(wavefuncs)):
                    for n in range(len(wavefuncs)):
                        for p in range(len(wavefuncs)):
                            for q in range(len(wavefuncs)):
                                s_w = c_mnpq[m, n, p, q] * j_op_1[m, n] * j_op_2[q, p]
                                if m == n and p == q:
                                    s_weight_conv[i, j] += s_w
                                else:
                                    s_weight_geom[i, j] += s_w

    return s_weight_conv, s_weight_geom


def _c_factor(e: npt.NDArray, psi: npt.NDArray, beta: float) -> npt.NDArray:
    n = len(e)
    c = np.zeros((n, n, n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            if np.isclose(e[i], e[j]):
                f_term = -_fermi_dirac_derivative(e[i], beta)
            else:
                f_term = (_fermi_dirac(e[i], beta) - _fermi_dirac(e[j], beta)) / (e[i] - e[j])
            for m in range(n):
                for n_ in range(n):
                    for p in range(n):
                        for q in range(n):
                            c[m, n_, p, q] += f_term * (
                                psi[:, i].conj()[m]
                                * psi[:, j][n_]
                                * psi[:, j].conj()[p]
                                * psi[:, i][q]
                            )
    return 2 * c


def _current_operator(h_deriv: npt.NDArray, psi: npt.NDArray) -> npt.NDArray:
    return psi.conj().T @ h_deriv @ psi


def _fermi_dirac(energy: float, beta: float) -> float:
    return 1.0 / (np.exp(beta * energy) + 1.0)


def _fermi_dirac_derivative(energy: float, beta: float) -> float:
    f = _fermi_dirac(energy, beta)
    return -beta * f * (1 - f)
