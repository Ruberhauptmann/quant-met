import numpy as np
from scipy import optimize

from quant_met.plotting.plot import scatter_into_BZ

from .configuration import Configuration, DeltaVector
from .gap_equation import gap_equation_real
from .nonint import generate_bloch


def solve_gap_equation():
    config = Configuration(
        a=np.sqrt(3) * 1, t_gr=1, t_x=0.01, V=1, U_X=1, U_Gr=1, mu=-1
    )

    all_K_points = (
        4
        * np.pi
        / (3 * config.a)
        * np.array(
            [
                (np.sin(i * np.pi / 6), np.cos(i * np.pi / 6))
                for i in [1, 3, 5, 7, 9, 11]
            ]
        )
    )
    K_vector_1 = all_K_points[1]
    K_vector_2 = all_K_points[5]

    nx = 5
    number_of_rows = 5

    k_points = np.concatenate(
        [
            np.linspace(
                i / (number_of_rows - 1) * K_vector_2,
                K_vector_1 + i / (number_of_rows - 1) * K_vector_2,
                num=nx,
            )
            for i in range(number_of_rows)
        ]
    )

    energies, bloch_absolute = generate_bloch(k_points, config)

    # plot_into_BZ(all_K_points, k_points)

    delta_vector = DeltaVector(k_points=k_points, initial=1)

    beta = 10000

    solution = optimize.fixed_point(
        gap_equation_real,
        delta_vector.as_1d_vector,
        args=(config.U, beta, bloch_absolute, energies),
    )

    delta_vector.update_from_1d_vector(solution)

    print(delta_vector)

    scatter_into_BZ(all_K_points, k_points, delta_vector.data.loc[:, "delta_3"])
