from typing import Any, List, Tuple

import matplotlib.axes
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection
from numpy import dtype, generic, ndarray


def scatter_into_bz(
    bz_corners: List[npt.NDArray[np.float64]],
    k_points: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64] | None = None,
    data_label: str | None = None,
    fig_in: matplotlib.figure.Figure | None = None,
    ax_in: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    if fig_in is None or ax_in is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_in, ax_in

    if data is not None:
        scatter = ax.scatter(*zip(*k_points), c=data, cmap="viridis")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label=data_label)
    else:
        ax.scatter(*zip(*k_points))

    ax.scatter(*zip(*bz_corners), alpha=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$k_x\ [1/a_0]$")
    ax.set_ylabel(r"$k_y\ [1/a_0]$")

    return fig


def plot_bandstructure(
    bands: npt.NDArray[np.float64],
    k_point_list: npt.NDArray[np.float64],
    ticks: List[float],
    labels: List[str],
    overlaps: npt.NDArray[np.float64] | None = None,
    overlap_labels: List[str] | None = None,
    fig_in: matplotlib.figure.Figure | None = None,
    ax_in: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    if fig_in is None or ax_in is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_in, ax_in

    ax.axhline(y=0, alpha=0.7, linestyle="--", color="black")

    if overlaps is None:
        for band in bands:
            ax.plot(k_point_list, band)
    else:
        line = LineCollection(segments=[np.array([(0, 0)])])
        for band, wx in zip(bands, overlaps):
            points = np.array([k_point_list, band]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            norm = matplotlib.colors.Normalize(-1, 1)
            lc = LineCollection(segments, cmap="seismic", norm=norm)
            lc.set_array(wx)
            lc.set_linewidth(2)
            line = ax.add_collection(lc)

        colorbar = fig.colorbar(line, fraction=0.046, pad=0.04, ax=ax)
        color_ticks = [-1, 1]
        colorbar.set_ticks(ticks=color_ticks, labels=overlap_labels)

    ax.set_ylim(
        top=np.max(bands) + 0.1 * np.max(bands),
        bottom=np.min(bands) - 0.1 * np.abs(np.min(bands)),
    )
    ax.set_box_aspect(1)
    ax.set_xticks(ticks, labels)
    ax.set_ylabel(r"$E\ [t]$")
    ax.set_facecolor("lightgray")
    ax.grid(visible=True)
    ax.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )

    return fig


def _generate_part_of_path(
    p_0: npt.NDArray[np.float64],
    p_1: npt.NDArray[np.float64],
    n: int,
    length_whole_path: int,
) -> npt.NDArray[np.float64]:
    distance = np.linalg.norm(p_1 - p_0)
    number_of_points = int(n * distance / length_whole_path) + 1

    k_space_path = np.vstack(
        [
            np.linspace(p_0[0], p_1[0], number_of_points),
            np.linspace(p_0[1], p_1[1], number_of_points),
        ]
    ).T[:-1]

    return k_space_path


def generate_bz_path(
    points: List[Tuple[npt.NDArray[np.float64], str]], number_of_points: int = 1000
) -> tuple[
    ndarray[Any, dtype[generic | generic | Any]],
    ndarray[Any, dtype[generic | generic | Any]],
    list[int | Any],
    list[str],
]:
    n = number_of_points

    cycle = [
        np.linalg.norm(points[i][0] - points[i + 1][0]) for i in range(len(points) - 1)
    ]
    cycle.append(np.linalg.norm(points[-1][0] - points[0][0]))

    length_whole_path = np.sum(np.array([cycle]))

    ticks = [0]
    for i in range(0, len(cycle) - 1):
        ticks.append(np.sum(cycle[0 : i + 1]) / length_whole_path)
    ticks.append(1)
    labels = [rf"${points[i][1]}$" for i in range(len(points))]
    labels.append(rf"${points[0][1]}$")

    whole_path_plot = np.concatenate(
        [
            np.linspace(
                ticks[i],
                ticks[i + 1],
                num=int(n * cycle[i] / length_whole_path),
                endpoint=False,
            )
            for i in range(0, len(ticks) - 1)
        ]
    )

    points_path = [
        _generate_part_of_path(points[i][0], points[i + 1][0], n, length_whole_path)
        for i in range(0, len(points) - 1)
    ]
    points_path.append(
        _generate_part_of_path(points[-1][0], points[0][0], n, length_whole_path)
    )
    whole_path = np.concatenate(points_path)

    return whole_path, whole_path_plot, ticks, labels
