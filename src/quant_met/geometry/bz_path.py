# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Methods to generate paths through the BZ."""

import numpy as np
import numpy.typing as npt


def _generate_part_of_path(
    p_0: npt.NDArray[np.float64],
    p_1: npt.NDArray[np.float64],
    n: int,
    length_whole_path: int,
) -> npt.NDArray[np.float64]:
    distance = np.linalg.norm(p_1 - p_0)
    number_of_points = int(n * distance / length_whole_path) + 1

    return np.vstack(
        [
            np.linspace(p_0[0], p_1[0], number_of_points),
            np.linspace(p_0[1], p_1[1], number_of_points),
        ]
    ).T[:-1]


def generate_bz_path(
    points: list[tuple[npt.NDArray[np.float64], str]], number_of_points: int = 1000
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    list[float],
    list[str],
]:
    """Generate a path through high symmetry points.

    Parameters
    ----------
    points : :class:`numpy.ndarray`
        Test
    number_of_points: int
        Number of point in the whole path.

    Returns
    -------
    :class:`numpy.ndarray`
        List of two-dimensional k points.
    :class:`numpy.ndarray`
        Path for plotting purposes: points between 0 and 1, with appropriate spacing.
    list[float]
        A list of ticks for the plotting path.
    list[str]
        A list of labels for the plotting path.

    """
    n = number_of_points

    cycle = [np.linalg.norm(points[i][0] - points[i + 1][0]) for i in range(len(points) - 1)]
    cycle.append(np.linalg.norm(points[-1][0] - points[0][0]))

    length_whole_path = np.sum(np.array([cycle]))

    ticks = [0.0]
    ticks.extend([np.sum(cycle[0 : i + 1]) / length_whole_path for i in range(len(cycle) - 1)])
    ticks.append(1.0)
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
            for i in range(len(ticks) - 1)
        ]
    )

    points_path = [
        _generate_part_of_path(points[i][0], points[i + 1][0], n, length_whole_path)
        for i in range(len(points) - 1)
    ]
    points_path.append(_generate_part_of_path(points[-1][0], points[0][0], n, length_whole_path))
    whole_path = np.concatenate(points_path)

    return whole_path, whole_path_plot, ticks, labels
