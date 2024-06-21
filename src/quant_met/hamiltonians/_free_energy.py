import numpy as np
import numpy.typing as npt

from ._base_hamiltonian import BaseHamiltonian


def free_energy(
    delta_vector: npt.NDArray[np.float64],
    hamiltonian: BaseHamiltonian,
    k_points: npt.NDArray[np.float64],
    beta: float | None = None,
) -> float:
    number_k_points = len(k_points)
    bdg_energies, bdg_vectors = hamiltonian.diagonalize_bdg(k_points, delta_vector)

    k_array: npt.NDArray[np.float64] = np.real(
        np.trace(hamiltonian.hamiltonian_k_space(k_points), axis1=-2, axis2=-1)
    ) + np.ones(number_k_points) * np.sum(
        np.power(np.abs(delta_vector), 2) / hamiltonian.coloumb_orbital_basis
    )
    if beta is None:
        k_array -= 0.5 * np.array(
            [
                np.real(
                    np.trace(
                        bdg_vectors[k_index]
                        @ np.diagflat(np.abs(bdg_energies[k_index]))
                        @ np.conjugate(bdg_vectors[k_index]).T
                    )
                )
                for k_index in range(number_k_points)
            ]
        )
    else:
        k_array -= (
            np.sum(np.log(1 + np.nan_to_num(np.exp(-beta * bdg_energies))), axis=-1)
            / beta
        )
    integral: float = np.sum(k_array, axis=-1) / (2.5980762113533156 * number_k_points)

    return integral


def free_energy_uniform_pairing(
    delta: float,
    hamiltonian: BaseHamiltonian,
    k_points: npt.NDArray[np.float64],
    beta: float | None = None,
) -> float:
    delta_vector = np.ones(hamiltonian.number_of_bands) * delta

    return free_energy(
        delta_vector=delta_vector, hamiltonian=hamiltonian, k_points=k_points, beta=beta
    )
