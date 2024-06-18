import numpy as np

from ._base_hamiltonian import BaseHamiltonian


def calculate_superfluid_weight(
    hamiltonian: BaseHamiltonian, direction_1, direction_2, k_points
):
    pass


def calculate_current_op(hamiltonian: BaseHamiltonian, direction, k_points):
    number_k_points = len(k_points)

    energies, bloch = hamiltonian.generate_bloch(k_points)

    eps = 1e-8
    # eps = 0

    if direction == 0:
        direction_vector = np.array([1, 0])
    else:
        direction_vector = np.array([0, 1])

    current_op = np.zeros(shape=(number_k_points, 3, 3))

    for k_index, energy in enumerate(energies):
        k = k_points[k_index]
        k_step = k + direction_vector * eps
        energy_step, bloch_step = hamiltonian.generate_bloch(np.array([k_step]))
        energy_step = energy_step[0]
        bloch_step = bloch_step[0]

        for m, n in np.ndindex(3, 3):
            bloch_prime = (bloch_step[m] - bloch[k_index][m]) / eps

            current_op[k_index, m, n] = (energy[m] - energy[n]) * np.dot(
                bloch_prime, bloch[k_index][n]
            )

        energy_prime = (energy_step - energy) / eps

        current_op[k_index] += np.eye(3) * energy_prime

    return current_op


def calculate_C_matrix():
    pass
