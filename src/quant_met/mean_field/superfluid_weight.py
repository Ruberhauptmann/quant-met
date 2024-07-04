# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Functions to calculate the superfluid weight."""

import numpy as np
import numpy.typing as npt

from .base_hamiltonian import BaseHamiltonian


def superfluid_weight(
    h: BaseHamiltonian,
    k_grid: npt.NDArray[np.float64],
    direction_1: str,
    direction_2: str,
) -> tuple[float, float]:
    """Calculate the superfluid weight.

    Parameters
    ----------
    h : :class:`~quant_met.mean_field.Hamiltonian`
        Hamiltonian.
    k_grid : :class:`numpy.ndarray`
        List of k points.
    direction_1 : str
        Direction 1, either 'x' or 'y'.
    direction_2
        Direction 2, either 'x' or 'y'.

    Returns
    -------
    float
        Conventional contribution to the superfluid weight.
    float
        Geometric contribution to the superfluid weight.

    """
    s_weight_conv = 0
    s_weight_geom = 0

    for k in k_grid:
        c_mnpq = _c_factor(h, k)
        j_up = _current_operator(h, direction_1, k)
        j_down = _current_operator(h, direction_2, -k)
        for m in range(h.number_of_bands):
            for n in range(h.number_of_bands):
                for p in range(h.number_of_bands):
                    for q in range(h.number_of_bands):
                        s_weight = c_mnpq[m, n, p, q] * j_up[m, n] * j_down[q, p]
                        if m == n and p == q:
                            s_weight_conv += s_weight
                        else:
                            s_weight_geom += s_weight

    return s_weight_conv, s_weight_geom


def _current_operator(
    h: BaseHamiltonian, direction: str, k: npt.NDArray[np.float64]
) -> npt.NDArray[np.complex64]:
    j = np.zeros(shape=(h.number_of_bands, h.number_of_bands), dtype=np.complex64)

    _, bloch = h.diagonalize_nonint(k=k)

    for m in range(h.number_of_bands):
        for n in range(h.number_of_bands):
            j[m, n] = (
                np.conjugate(bloch[:, m])
                @ h.hamiltonian_derivative(direction=direction, k=k)
                @ bloch[:, n]
            )

    return j


def _w_matrix(
    h: BaseHamiltonian, k: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.complex64], npt.NDArray[np.complex64]]:
    _, bloch = h.diagonalize_nonint(k=k)
    _, bdg_functions = h.diagonalize_bdg(k=k)

    w_plus = np.zeros((2 * h.number_of_bands, h.number_of_bands), dtype=np.complex64)
    for i in range(2 * h.number_of_bands):
        for m in range(h.number_of_bands):
            w_plus[i, m] = (
                np.tensordot(bloch[:, m], np.array([1, 0]), axes=0).reshape(-1)
                @ bdg_functions[:, i]
            )

    w_minus = np.zeros((2 * h.number_of_bands, h.number_of_bands), dtype=np.complex64)
    for i in range(2 * h.number_of_bands):
        for m in range(h.number_of_bands):
            w_minus[i, m] = (
                np.tensordot(np.conjugate(bloch[:, m]), np.array([0, 1]), axes=0).reshape(-1)
                @ bdg_functions[:, i]
            )

    return w_plus, w_minus


def _c_factor(h: BaseHamiltonian, k: npt.NDArray[np.float64]) -> npt.NDArray[np.complex64]:
    bdg_energies, _ = h.diagonalize_bdg(k)
    w_plus, w_minus = _w_matrix(h, k)
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
                                np.conjugate(w_minus[i, m])
                                * w_plus[j, n]
                                * np.conjugate(w_minus[j, p])
                                * w_minus[i, q]
                            )

                    c_mnpq[m, n, p, q] = 2 * c_tmp

    return c_mnpq


def _fermi_dirac_derivative() -> float:
    return 0


def _fermi_dirac(energy: np.float64) -> np.float64:
    if energy > 0:
        return np.float64(0)

    return np.float64(1)
