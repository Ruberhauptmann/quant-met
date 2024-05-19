import numpy as np
import numpy.typing as npt
from scipy import interpolate, optimize

from quant_met.configuration import Configuration, DeltaVector

from .gap_equation_old import gap_equation_real
from .nonint_old import generate_bloch


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


def solve_gap_equation(config: Configuration, k_points: npt.NDArray) -> DeltaVector:
    energies, bloch_absolute = generate_bloch(k_points, config)

    delta_vector = DeltaVector(k_points=k_points, initial=1)

    solution = optimize.fixed_point(
        gap_equation_real,
        delta_vector.as_1d_vector,
        args=(config.U, config.beta, bloch_absolute, energies, config.mu),
    )

    delta_vector.update_from_1d_vector(solution)

    return delta_vector


def interpolate_gap(
    delta_vector_on_grid: DeltaVector, bandpath: npt.NDArray
) -> DeltaVector:
    delta_vector_interpolated = DeltaVector(k_points=bandpath)

    for band in [1, 2, 3]:
        delta_vector_interpolated.data.loc[:, f"delta_{band}"] = interpolate.griddata(
            delta_vector_on_grid.k_points,
            delta_vector_on_grid.data.loc[:, f"delta_{band}"],
            bandpath,
            method="cubic",
        )

    return delta_vector_interpolated
