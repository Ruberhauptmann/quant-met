# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Functions to calculate the superfluid weight."""

import numpy as np
import numpy.typing as npt

from quant_met.mean_field.hamiltonians.base_hamiltonian import BaseHamiltonian
from quant_met.parameters import GenericParameters


def superfluid_weight(
    h: BaseHamiltonian[GenericParameters],
    k: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.complex64], npt.NDArray[np.complex64]]:
    """Calculate the superfluid weight.

    Parameters
    ----------
    h : :class:`~quant_met.mean_field.Hamiltonian`
        Hamiltonian.
    k : :class:`numpy.ndarray`
        List of k points.

    Returns
    -------
    :class:`numpy.ndarray`
        Conventional contribution to the superfluid weight.
    :class:`numpy.ndarray`
        Geometric contribution to the superfluid weight.

    """
    s_weight_conv = np.zeros(shape=(2, 2), dtype=np.complex64)
    s_weight_geom = np.zeros(shape=(2, 2), dtype=np.complex64)

    for i, direction_1 in enumerate(["x", "y"]):
        for j, direction_2 in enumerate(["x", "y"]):
            for k_point in k:
                c_mnpq = _c_factor(h, k_point)
                j_up = _current_operator(h, direction_1, k_point)
                j_down = _current_operator(h, direction_2, -k_point)
                for m in range(h.number_of_bands):
                    for n in range(h.number_of_bands):
                        for p in range(h.number_of_bands):
                            for q in range(h.number_of_bands):
                                s_weight = c_mnpq[m, n, p, q] * j_up[m, n] * j_down[q, p]
                                if m == n and p == q:
                                    s_weight_conv[i, j] += s_weight
                                else:
                                    s_weight_geom[i, j] += s_weight

    return s_weight_conv, s_weight_geom


def _current_operator(
    h: BaseHamiltonian[GenericParameters], direction: str, k: npt.NDArray[np.float64]
) -> npt.NDArray[np.complex64]:
    j = np.zeros(shape=(h.number_of_bands, h.number_of_bands), dtype=np.complex64)

    _, bloch = h.diagonalize_nonint(k=k)

    for m in range(h.number_of_bands):
        for n in range(h.number_of_bands):
            j[m, n] = (
                bloch[:, m].conjugate()
                @ h.hamiltonian_derivative(direction=direction, k=k)
                @ bloch[:, n]
            )

    return j


def _c_factor(
    h: BaseHamiltonian[GenericParameters], k: npt.NDArray[np.float64]
) -> npt.NDArray[np.complex64]:
    bdg_energies, bdg_functions = h.diagonalize_bdg(k)
    c_mnpq = np.zeros(
        shape=(
            h.number_of_bands,
            h.number_of_bands,
            h.number_of_bands,
            h.number_of_bands,
        ),
        dtype=np.complex64,
    )

    for m in range(h.number_of_bands):
        for n in range(h.number_of_bands):
            for p in range(h.number_of_bands):
                for q in range(h.number_of_bands):
                    c_tmp: float = 0
                    for i in range(2 * h.number_of_bands):
                        for j in range(2 * h.number_of_bands):
                            if bdg_energies[i] != bdg_energies[j]:
                                c_tmp += (
                                    _fermi_dirac(bdg_energies[i]) - _fermi_dirac(bdg_energies[j])
                                ) / (bdg_energies[j] - bdg_energies[i])
                            else:
                                c_tmp -= _fermi_dirac_derivative()

                            c_tmp *= (
                                bdg_functions[i, m].conjugate()
                                * bdg_functions[j, n]
                                * bdg_functions[j, p].conjugate()
                                * bdg_functions[i, q].conjugate()
                            )

                    c_mnpq[m, n, p, q] = 2 * c_tmp

    return c_mnpq


def _fermi_dirac_derivative() -> float:
    return 0


def _fermi_dirac(energy: np.float64) -> np.float64:
    if energy > 0:
        return np.float64(0)

    return np.float64(1)
