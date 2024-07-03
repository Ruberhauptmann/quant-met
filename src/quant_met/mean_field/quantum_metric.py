# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

import numpy as np
import numpy.typing as npt

from .base_hamiltonian import BaseHamiltonian


def quantum_metric(
    h: BaseHamiltonian, k_grid: npt.NDArray[np.float64], band: int
) -> npt.NDArray[np.float64]:
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
                            np.conjugate(bloch[k_index][:, band])
                            @ h_derivative_direction_1[k_index]
                            @ bloch[k_index][:, n]
                        )
                        * (
                            np.conjugate(bloch[k_index][:, n])
                            @ h_derivative_direction_2[k_index]
                            @ bloch[k_index][:, band]
                        )
                        / (energies[k_index][band] - energies[k_index][n]) ** 2
                    )

    return np.real(quantum_geom_tensor) / number_k_points
