import numpy as np
import numpy.typing as npt
from scipy import optimize

from quant_met.hamiltonians import BaseHamiltonian


def free_energy(
    delta_vector: npt.NDArray,
    beta: float,
    hamiltonian: BaseHamiltonian,
    k_points: npt.NDArray,
) -> float:
    number_k_points = len(k_points)
    bdg_energies, _ = hamiltonian.diagonalize_bdg(k_points, delta_vector)
    """
    print(bdg_energies)
    print(np.power(np.abs(delta_vector), 2) / hamiltonian.coloumb_orbital_basis)
    print(np.sum(np.power(np.abs(delta_vector), 2) / hamiltonian.coloumb_orbital_basis))
    print(
        np.ones(number_k_points)
        * np.sum(np.power(np.abs(delta_vector), 2) / hamiltonian.coloumb_orbital_basis)
    )
    """
    k_array = (
        np.real(np.trace(hamiltonian.hamiltonian_k_space(k_points), axis1=-2, axis2=-1))
        + np.ones(number_k_points)
        * np.sum(np.power(np.abs(delta_vector), 2) / hamiltonian.coloumb_orbital_basis)
        - np.sum(np.log(1 + np.nan_to_num(np.exp(-beta * bdg_energies))), axis=-1)
        / beta
    )

    return np.sum(k_array) / (2.5980762113533156 * number_k_points)


def minimize_loop(
    beta: float, hamiltonian: BaseHamiltonian, k_points: npt.NDArray
) -> optimize.OptimizeResult:
    initial_guess = np.ones(shape=hamiltonian.number_of_bands)
    solution = optimize.differential_evolution(
        free_energy,
        tol=1e-12,
        x0=initial_guess,
        atol=0,
        # options={"eps": 1e-12, 'ftol': 1e-12, 'gtol': 1e-12},
        args=(beta, hamiltonian, k_points),
        bounds=[(0, 5) for _ in range(hamiltonian.number_of_bands)],
    )

    return solution
