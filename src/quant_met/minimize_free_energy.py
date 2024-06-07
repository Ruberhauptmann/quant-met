import numpy as np
import numpy.typing as npt
from scipy import optimize

from quant_met.hamiltonians import BaseHamiltonian


def free_energy(
    delta_vector: npt.NDArray[np.float64],
    beta: float,
    hamiltonian: BaseHamiltonian,
    k_points: npt.NDArray[np.float64],
) -> float:
    number_k_points = len(k_points)
    bdg_energies, _ = hamiltonian.diagonalize_bdg(k_points, delta_vector)
    k_array: npt.NDArray[np.float64] = (
        np.real(np.trace(hamiltonian.hamiltonian_k_space(k_points), axis1=-2, axis2=-1))
        + np.ones(number_k_points)
        * np.sum(np.power(np.abs(delta_vector), 2) / hamiltonian.coloumb_orbital_basis)
        - np.sum(np.log(1 + np.nan_to_num(np.exp(-beta * bdg_energies))), axis=-1)
        / beta
    )
    integral: float = np.sum(k_array, axis=0) / (2.5980762113533156 * number_k_points)

    return integral


def minimize_loop(
    beta: float, hamiltonian: BaseHamiltonian, k_points: npt.NDArray[np.float64]
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
