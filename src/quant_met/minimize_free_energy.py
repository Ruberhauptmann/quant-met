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
    sum_tmp = 0

    for k in k_points:
        sum_tmp += np.real(np.trace(hamiltonian.hamiltonian_k_space(k)[0]))
        sum_tmp -= np.sum(
            np.power(np.abs(delta_vector), 2) / hamiltonian.coloumb_orbital_basis
        )
        bdg_energies, _ = hamiltonian.diagonalize_bdg(k, delta_vector)
        sum_tmp -= np.sum(np.log(1 + np.exp(-beta * bdg_energies))) / beta

    return sum_tmp / (2.5980762113533156 * number_k_points)


def write_progress(intermediate_result: optimize.OptimizeResult):
    print(np.linalg.norm(intermediate_result.x))


def minimize_loop(
    beta: float, hamiltonian: BaseHamiltonian, k_points: npt.NDArray
) -> optimize.OptimizeResult:
    initial_guess = np.ones(shape=hamiltonian.number_of_bands) * 0.1
    solution = optimize.minimize(
        free_energy,
        tol=1e-12,
        options={"disp": True},
        x0=initial_guess,
        args=(beta, hamiltonian, k_points),
        bounds=[(0, 10) for _ in range(hamiltonian.number_of_bands)],
        callback=write_progress,
    )

    return solution
