import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate, optimize

from quant_met import plotting
from quant_met.configuration import Configuration, DeltaVector

from .gap_equation import gap_equation_real
from .nonint import generate_bloch


def solve_gap_equation():
    config = Configuration(
        a=np.sqrt(3) * 1, t_gr=1, t_x=0.01, V=0.5, U_X=0, U_Gr=0, mu=-2
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

    nx = 8
    number_of_rows = 8

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

    plotting.plot_into_BZ(all_K_points, k_points)

    delta_vector = DeltaVector(k_points=k_points, initial=1)

    beta = 100000

    solution = optimize.fixed_point(
        gap_equation_real,
        delta_vector.as_1d_vector,
        args=(config.U, beta, bloch_absolute, energies),
    )

    delta_vector.update_from_1d_vector(solution)

    delta_vector.save("gap.hdf5")

    plotting.scatter_into_BZ(
        all_K_points, k_points, delta_vector.data.loc[:, "delta_3"]
    )

    whole_path, whole_path_plot, ticks, labels = plotting.generate_BZ_path(config.a)

    gap1_on_bandpath = np.abs(
        interpolate.griddata(
            delta_vector.k_points,
            delta_vector.data.loc[:, "delta_1"],
            whole_path,
            method="cubic",
        )
    )
    gap2_on_bandpath = np.abs(
        interpolate.griddata(
            delta_vector.k_points,
            delta_vector.data.loc[:, "delta_2"],
            whole_path,
            method="cubic",
        )
    )
    gap3_on_bandpath = np.abs(
        interpolate.griddata(
            delta_vector.k_points,
            delta_vector.data.loc[:, "delta_3"],
            whole_path,
            method="cubic",
        )
    )

    # scatter_into_BZ(all_K_points, whole_path, gap1_on_bandpath)

    fig, ax = plt.subplots()

    energies_on_bandpath, _ = generate_bloch(whole_path, config)

    ax.plot(
        whole_path_plot,
        np.sqrt(energies_on_bandpath[:, 0] ** 2 + gap1_on_bandpath**2),
        label="band 1, +",
    )
    ax.plot(
        whole_path_plot,
        -np.sqrt(energies_on_bandpath[:, 0] ** 2 + gap1_on_bandpath**2),
        label="band 1, -",
    )
    ax.plot(
        whole_path_plot,
        np.sqrt(energies_on_bandpath[:, 1] ** 2 + gap2_on_bandpath**2),
        label="band 2, +",
    )
    ax.plot(
        whole_path_plot,
        -np.sqrt(energies_on_bandpath[:, 1] ** 2 + gap2_on_bandpath**2),
        label="band 2, -",
    )
    ax.plot(
        whole_path_plot,
        np.sqrt(energies_on_bandpath[:, 2] ** 2 + gap3_on_bandpath**2),
        label="band 3, +",
    )
    ax.plot(
        whole_path_plot,
        -np.sqrt(energies_on_bandpath[:, 2] ** 2 + gap3_on_bandpath**2),
        label="band 3, -",
    )

    ax.axhline(y=config.mu)

    ax.set_xticks(ticks, labels)
    ax.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )

    fig.legend()

    fig.savefig("BCS_bandstructure.pdf", bbox_inches="tight")
