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
    # print(np.sum(np.power(np.abs(delta_vector), 2) / hamiltonian.coloumb_orbital_basis))
    # print(np.real(np.trace(hamiltonian.hamiltonian_k_space(k_points), axis1=-2, axis2=-1)))
    # print(bdg_energies)
    # print(np.sum(bdg_energies, axis=-1))
    k_array: npt.NDArray[np.float64] = (
        np.real(np.trace(hamiltonian.hamiltonian_k_space(k_points), axis1=-2, axis2=-1))
        + np.ones(number_k_points)
        * np.sum(np.power(np.abs(delta_vector), 2) / hamiltonian.coloumb_orbital_basis)
        # - 0.5 * np.sum(bdg_energies, axis=-1)
        - np.sum(np.log(1 + np.nan_to_num(np.exp(-beta * bdg_energies))), axis=-1)
        / beta
    )
    # print(k_array)
    # print(np.sum(k_array))
    integral: float = np.sum(k_array) / (2.5980762113533156 * number_k_points)

    return integral


def minimize_loop(
    beta: float, hamiltonian: BaseHamiltonian, k_points: npt.NDArray[np.float64]
) -> optimize.OptimizeResult:
    initial_guess = np.ones(shape=hamiltonian.number_of_bands) * 5
    solution = optimize.brute(
        func=free_energy,
        # tol=1e-12,
        # x0=initial_guess,
        # atol=0,
        # options={"eps": 1e-12, 'ftol': 1e-12, 'gtol': 1e-12},
        args=(beta, hamiltonian, k_points),
        ranges=[(0, 1) for _ in range(hamiltonian.number_of_bands)],
        Ns=20,
        workers=10,
        finish=optimize.fmin,
        # bounds=[(0, 2) for _ in range(hamiltonian.number_of_bands)],
    )

    return solution
