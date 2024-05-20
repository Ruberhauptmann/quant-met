import numpy as np
import numpy.typing as npt
from scipy import interpolate, optimize

from quant_met.bcs.gap_equation import gap_equation_real
from quant_met.configuration import DeltaVector
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
    hamiltonian: BaseHamiltonian, k_points: npt.NDArray, beta: float = 0
) -> DeltaVector:
    energies, bloch_absolute = hamiltonian.generate_bloch(k_points=k_points)

    delta_vector = DeltaVector(
        k_points=k_points, initial=0.1, number_bands=hamiltonian.number_bands
    )
    try:
        solution = optimize.fixed_point(
            gap_equation_real,
            delta_vector.as_1d_vector,
            args=(hamiltonian.U, beta, bloch_absolute, energies, len(k_points)),
        )
    except RuntimeError:
        print("Failed")
        solution = DeltaVector(
            k_points=k_points, initial=0.0, number_bands=hamiltonian.number_bands
        ).as_1d_vector

    delta_vector.update_from_1d_vector(solution)

    return delta_vector


def interpolate_gap(
    delta_vector_on_grid: DeltaVector, bandpath: npt.NDArray
) -> DeltaVector:
    delta_vector_interpolated = DeltaVector(
        k_points=bandpath, number_bands=delta_vector_on_grid.number_bands
    )

    for band in range(delta_vector_interpolated.number_bands):
        delta_vector_interpolated.data.loc[:, f"delta_{band}"] = interpolate.griddata(
            delta_vector_on_grid.k_points,
            delta_vector_on_grid.data.loc[:, f"delta_{band}"],
            bandpath,
            method="cubic",
        )

    return delta_vector_interpolated
