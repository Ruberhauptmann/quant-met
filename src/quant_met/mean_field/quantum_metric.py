# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Functions to calculate the quantum metric."""

import numpy as np
import numpy.typing as npt

from .base_hamiltonian import BaseHamiltonian


def quantum_metric(
    h: BaseHamiltonian, k_grid: npt.NDArray[np.float64], band: int
) -> npt.NDArray[np.float64]:
    """Calculate the quantum metric in the normal state.

    Parameters
    ----------
    h : :class:`~quant_met.BaseHamiltonian`
        Hamiltonian object.
    k_grid : :class:`numpy.ndarray`
        List of k points.
    band : int
        Index of band for which the quantum metric is calculated.

    Returns
    -------
    :class:`numpy.ndarray`
        Quantum metric in the normal state.

    """
    energies, bloch = h.diagonalize_nonint(k_grid)

    number_k_points = len(k_grid)

    quantum_geom_tensor = np.zeros(shape=(2, 2), dtype=np.complex64)

    for i, direction_1 in enumerate(["x", "y"]):
        h_derivative_direction_1 = h.hamiltonian_derivative(k=k_grid, direction=direction_1)
        for j, direction_2 in enumerate(["x", "y"]):
            h_derivative_direction_2 = h.hamiltonian_derivative(k=k_grid, direction=direction_2)
            for k_index in range(len(k_grid)):
                for n in [i for i in range(h.number_of_bands) if i != band]:
                    quantum_geom_tensor[i, j] += (
                        (
                            bloch[k_index][:, band].conjugate()
                            @ h_derivative_direction_1[k_index]
                            @ bloch[k_index][:, n]
                        )
                        * (
                            bloch[k_index][:, n].conjugate()
                            @ h_derivative_direction_2[k_index]
                            @ bloch[k_index][:, band]
                        )
                        / (energies[k_index][band] - energies[k_index][n]) ** 2
                    )

    return np.real(quantum_geom_tensor) / number_k_points


def quantum_metric_bdg(
    h: BaseHamiltonian, k_grid: npt.NDArray[np.float64], band: int
) -> npt.NDArray[np.float64]:
    """Calculate the quantum metric in the BdG state.

    Parameters
    ----------
    h : :class:`~quant_met.BaseHamiltonian`
        Hamiltonian object.
    k_grid : :class:`numpy.ndarray`
        List of k points.
    band : int
        Index of band for which the quantum metric is calculated.

    Returns
    -------
    :class:`numpy.ndarray`
        Quantum metric in the normal state.

    """
    energies, bdg_functions = h.diagonalize_bdg(k_grid)

    number_k_points = len(k_grid)

    quantum_geom_tensor = np.zeros(shape=(2, 2), dtype=np.complex64)

    for i, direction_1 in enumerate(["x", "y"]):
        h_derivative_direction_1 = h.bdg_hamiltonian_derivative(k=k_grid, direction=direction_1)
        for j, direction_2 in enumerate(["x", "y"]):
            h_derivative_direction_2 = h.bdg_hamiltonian_derivative(k=k_grid, direction=direction_2)
            for k_index in range(len(k_grid)):
                for n in [i for i in range(h.number_of_bands) if i != band]:
                    quantum_geom_tensor[i, j] += (
                        (
                            bdg_functions[k_index][:, band].conjugate()
                            @ h_derivative_direction_1[k_index]
                            @ bdg_functions[k_index][:, n]
                        )
                        * (
                            bdg_functions[k_index][:, n].conjugate()
                            @ h_derivative_direction_2[k_index]
                            @ bdg_functions[k_index][:, band]
                        )
                        / (energies[k_index][band] - energies[k_index][n]) ** 2
                    )

    return np.real(quantum_geom_tensor) / number_k_points


def quantum_metric_bdg_occ(
    h: BaseHamiltonian, k_grid: npt.NDArray[np.float64], band: int
) -> npt.NDArray[np.float64]:
    """Calculate the quantum metric in the BdG state.

    Parameters
    ----------
    h : :class:`~quant_met.BaseHamiltonian`
        Hamiltonian object.
    k_grid : :class:`numpy.ndarray`
        List of k points.
    band : int
        Index of band for which the quantum metric is calculated.

    Returns
    -------
    :class:`numpy.ndarray`
        Quantum metric in the normal state.

    """
    energies, bdg_functions = h.diagonalize_bdg(k_grid)

    number_k_points = len(k_grid)

    quantum_geom_tensor = np.zeros(shape=(2, 2), dtype=np.complex64)

    for i, direction_1 in enumerate(["x", "y"]):
        h_derivative_direction_1 = h.bdg_hamiltonian_derivative(k=k_grid, direction=direction_1)
        for j, direction_2 in enumerate(["x", "y"]):
            h_derivative_direction_2 = h.bdg_hamiltonian_derivative(k=k_grid, direction=direction_2)
            for k_index in range(len(k_grid)):
                for n in [
                    i for i in range(h.number_of_bands) if i != band and energies[k_index][i] < 0
                ]:
                    quantum_geom_tensor[i, j] += (
                        (
                            bdg_functions[k_index][:, band].conjugate()
                            @ h_derivative_direction_1[k_index]
                            @ bdg_functions[k_index][:, n]
                        )
                        * (
                            bdg_functions[k_index][:, n].conjugate()
                            @ h_derivative_direction_2[k_index]
                            @ bdg_functions[k_index][:, band]
                        )
                        / (energies[k_index][band] - energies[k_index][n]) ** 2
                    )

    return np.real(quantum_geom_tensor) / number_k_points
