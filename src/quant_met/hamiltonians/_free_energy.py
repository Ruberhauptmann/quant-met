# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

import numpy as np
import numpy.typing as npt

from ._base_hamiltonian import BaseHamiltonian


def free_energy(
    delta_vector: npt.NDArray[np.float64],
    hamiltonian: BaseHamiltonian,
    k_points: npt.NDArray[np.float64],
) -> float:
    number_k_points = len(k_points)
    hamiltonian.delta_orbital_basis = delta_vector
    bdg_energies, _ = hamiltonian.diagonalize_bdg(k_points)

    k_array = np.array(
        [
            np.sum(np.abs(bdg_energies[k_index][0 : hamiltonian.number_of_bands]))
            for k_index in range(number_k_points)
        ]
    )

    integral: float = -np.sum(k_array, axis=-1) / number_k_points + np.sum(
        np.power(np.abs(delta_vector), 2) / hamiltonian.coloumb_orbital_basis
    )

    return integral


def free_energy_uniform_pairing(
    delta: float,
    hamiltonian: BaseHamiltonian,
    k_points: npt.NDArray[np.float64],
) -> float:
    delta_vector = np.ones(hamiltonian.number_of_bands) * delta

    return free_energy(
        delta_vector=delta_vector, hamiltonian=hamiltonian, k_points=k_points
    )
