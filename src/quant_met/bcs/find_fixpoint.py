import numpy as np
import numpy.typing as npt
from scipy import optimize

from quant_met.bcs.gap_equation import gap_equation_real
from quant_met.configuration import Configuration, DeltaVector
from quant_met.hamiltonians import BaseHamiltonian


def generate_k_space_grid(nx, nrows, corner_1, corner_2):
    k_points = np.concatenate(
        [
            np.linspace(
                i / (nrows - 1) * corner_2,
                corner_1 + i / (nrows - 1) * corner_2,
                num=nx,
            )
            for i in range(nrows)
        ]
    )

    return k_points


def solve_gap_equation(
    config: Configuration, hamiltonian: BaseHamiltonian, k_points: npt.NDArray
) -> DeltaVector:
    energies, bloch_absolute = hamiltonian.generate_bloch(
        k_points=k_points, mu=config.mu
    )

    delta_vector = DeltaVector(
        k_points=k_points, initial=0.1, number_bands=hamiltonian.number_bands
    )

    try:
        solution = optimize.fixed_point(
            gap_equation_real,
            delta_vector.as_1d_vector,
            args=(config.U, config.beta, bloch_absolute, energies, len(k_points)),
            # xtol=1e-10
        )
    except RuntimeError:
        print("Failed")
        solution = DeltaVector(
            k_points=k_points, initial=0.0, number_bands=hamiltonian.number_bands
        ).as_1d_vector

    delta_vector.update_from_1d_vector(solution)

    return delta_vector
