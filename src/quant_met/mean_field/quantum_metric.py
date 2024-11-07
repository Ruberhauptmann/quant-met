# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Functions to calculate the quantum metric."""

import numpy as np
import numpy.typing as npt

from quant_met.mean_field.hamiltonians.base_hamiltonian import BaseHamiltonian
from quant_met.parameters import GenericParameters


def quantum_metric(
    h: BaseHamiltonian[GenericParameters], k: npt.NDArray[np.float64], bands: list[int]
) -> npt.NDArray[np.float64]:
    """Calculate the quantum metric (geometric tensor) for specified bands.

    This function computes the quantum geometric tensor associated with
    the specified bands of a given Hamiltonian over a grid of k-points.
    The output is a 2x2 matrix representing the quantum metric.

    Parameters
    ----------
    h : BaseHamiltonian
        Hamiltonian object used to compute Bloch states and their derivatives.
    k : numpy.ndarray
        Array of k points in the Brillouin zone.
    bands : list of int
        Indices of the bands for which the quantum metric is to be calculated.

    Returns
    -------
    :class:`numpy.ndarray`
        A 2x2 matrix representing the quantum metric.

    Raises
    ------
    ValueError
        If `bands` contains invalid indices or `k_grid` is empty.
    """
    energies, bloch = h.diagonalize_nonint(k)

    number_k_points = len(k)

    quantum_geom_tensor = np.zeros(shape=(2, 2), dtype=np.complex64)

    for band in bands:
        for i, direction_1 in enumerate(["x", "y"]):
            h_derivative_direction_1 = h.hamiltonian_derivative(k=k, direction=direction_1)
            for j, direction_2 in enumerate(["x", "y"]):
                h_derivative_direction_2 = h.hamiltonian_derivative(k=k, direction=direction_2)
                for k_index in range(len(k)):
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
