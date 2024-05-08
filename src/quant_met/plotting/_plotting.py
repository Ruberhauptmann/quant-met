import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def plot_into_BZ(all_K_points, k_points):
    fig, ax = plt.subplots()

    ax.scatter(*zip(*k_points))
    ax.scatter(*zip(*all_K_points), alpha=0.6)

    ax.set_aspect("equal", adjustable="box")

    return fig


def scatter_into_BZ(all_K_points, k_points, data):
    fig, ax = plt.subplots()

    scatter = ax.scatter(*zip(*k_points), c=data, cmap="viridis")
    ax.scatter(*zip(*all_K_points), alpha=0.8)
    fig.colorbar(scatter, ax=ax, ticks=[0, 1, 2, 3])

    ax.set_aspect("equal", adjustable="box")

    return fig


def plot_nonint_bandstructure(
    bands, overlaps, k_point_list, ticks, labels, fig: plt.Figure, ax: plt.Axes
):
    line = None
    ax.axhline(y=0, alpha=0.7, linestyle="--", color="black")

    for band, wx in zip(bands, overlaps):
        points = np.array([k_point_list, band]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(-1, 1)
        lc = LineCollection(segments, cmap="seismic", norm=norm)
        lc.set_array(wx)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)

    colorbar = fig.colorbar(line, fraction=0.046, pad=0.04, ax=ax)
    color_ticks = [-1, 1]
    colorbar.set_ticks(ticks=color_ticks, labels=[r"$w_{\mathrm{Gr}_1}$", r"$w_X$"])

    ax.set_box_aspect(1)

    ax.set_xticks(ticks, labels)
    ax.set_yticks(range(-5, 6))
    # ax.set_ylim([-5, 5])
    ax.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )


def _generate_part_of_path(p_0, p_1, n, length_whole_path):
    distance = np.linalg.norm(p_1 - p_0)
    number_of_points = int(n * distance / length_whole_path) + 1

    k_space_path = np.vstack(
        [
            np.linspace(p_0[0], p_1[0], number_of_points),
            np.linspace(p_0[1], p_1[1], number_of_points),
        ]
    ).T[:-1]

    return k_space_path


def generate_BZ_path(a, points=None):
    Gamma = np.array([0, 0])
    M = np.pi / a * np.array([1, 1 / np.sqrt(3)])
    K = 4 * np.pi / (3 * a) * np.array([1, 0])

    n = 1000

    length_whole_path = (
        np.linalg.norm(M - Gamma) + np.linalg.norm(Gamma - K) + np.linalg.norm(K - M)
    )

    x_1 = np.linalg.norm(M - Gamma) / length_whole_path
    x_2 = (np.linalg.norm(M - Gamma) + np.linalg.norm(Gamma - K)) / length_whole_path

    whole_path_plot = np.concatenate(
        (
            np.linspace(
                0,
                x_1,
                num=int(n * np.linalg.norm(M - Gamma) / length_whole_path),
                endpoint=False,
            ),
            np.linspace(
                x_1,
                x_2,
                num=int(n * np.linalg.norm(Gamma - K) / length_whole_path),
                endpoint=False,
            ),
            np.linspace(
                x_2,
                1,
                num=int(n * np.linalg.norm(K - M) / length_whole_path),
                endpoint=False,
            ),
        )
    )

    ticks = [0, x_1, x_2, 1]
    labels = ["$M$", r"$\Gamma$", "$K$", "$M$"]

    whole_path = np.concatenate(
        (
            _generate_part_of_path(M, Gamma, n, length_whole_path),
            _generate_part_of_path(Gamma, K, n, length_whole_path),
            _generate_part_of_path(K, M, n, length_whole_path),
        )
    )

    return whole_path, whole_path_plot, ticks, labels
