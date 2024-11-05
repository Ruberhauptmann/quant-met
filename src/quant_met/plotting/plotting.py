# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Methods for plotting data."""

import matplotlib.axes
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.collections import Collection, LineCollection


def format_plot(
    ax: matplotlib.axes.Axes,
) -> matplotlib.axes.Axes:
    """Format the axis to the predefined style.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`

    Returns
    -------
    :class:`matplotlib.axes.Axes`

    """
    ax.set_box_aspect(1)
    ax.set_facecolor("lightgray")
    ax.grid(visible=True)
    ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)

    return ax


def scatter_into_bz(
    bz_corners: list[npt.NDArray[np.float64]],
    k_points: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64] | None = None,
    data_label: str | None = None,
    fig_in: matplotlib.figure.Figure | None = None,
    ax_in: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Scatter a list of points into the brillouin zone.

    Parameters
    ----------
    bz_corners : list[:class:`numpy.ndarray`]
        Corner points defining the brillouin zone.
    k_points : :class:`numpy.ndarray`
        List of k points.
    data : :class:`numpy.ndarray`, optional
        Data to put on the k points.
    data_label : :class:`str`, optional
        Label for the data.
    fig_in : :class:`matplotlib.figure.Figure`, optional
        Figure that holds the axes. If not provided, a new figure and ax is created.
    ax_in : :class:`matplotlib.axes.Axes`, optional
        Ax to plot the data in. If not provided, a new figure and ax is created.

    Returns
    -------
    :obj:`matplotlib.figure.Figure`
        Figure with the data plotted onto the axis.

    """
    if fig_in is None or ax_in is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_in, ax_in

    if data is not None:
        x_coords, y_coords = zip(*k_points, strict=True)
        scatter = ax.scatter(x=x_coords, y=y_coords, c=data, cmap="viridis")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label=data_label)
    else:
        x_coords, y_coords = zip(*k_points, strict=True)
        ax.scatter(x=x_coords, y=y_coords)

    bz_corner_x, bz_corners_y = zip(*bz_corners, strict=True)
    ax.scatter(x=bz_corner_x, y=bz_corners_y, alpha=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$k_x\ [1/a_0]$")
    ax.set_ylabel(r"$k_y\ [1/a_0]$")

    return fig


def plot_bandstructure(
    bands: npt.NDArray[np.float64],
    k_point_list: npt.NDArray[np.float64],
    ticks: list[float],
    labels: list[str],
    overlaps: npt.NDArray[np.float64] | None = None,
    overlap_labels: list[str] | None = None,
    fig_in: matplotlib.figure.Figure | None = None,
    ax_in: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot bands along a k space path.

    To have a plot that respects the distances in k space and generate everything that is needed for
    plotting, use the function :func:`~quant_met.plotting.generate_bz_path`.

    Parameters
    ----------
    bands : :class:`numpy.ndarray`
    k_point_list : :class:`numpy.ndarray`
        List of points to plot against. This is not a list of two-dimensional k-points!
    ticks : list(float)
        Position for ticks.
    labels : list(str)
        Labels on ticks.
    overlaps : :class:`numpy.ndarray`, optional
        Overlaps.
    overlap_labels : list(str), optional
        Labels to put on overlaps.
    fig_in : :class:`matplotlib.figure.Figure`, optional
        Figure that holds the axes. If not provided, a new figure and ax is created.
    ax_in : :class:`matplotlib.axes.Axes`, optional
        Ax to plot the data in. If not provided, a new figure and ax is created.

    Returns
    -------
    :obj:`matplotlib.figure.Figure`
        Figure with the data plotted onto the axis.

    """
    if fig_in is None or ax_in is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_in, ax_in

    ax.axhline(y=0, alpha=0.7, linestyle="--", color="black")

    if overlaps is None:
        for band in bands:
            ax.plot(k_point_list, band)
    else:
        line = Collection()
        for band, wx in zip(bands, overlaps, strict=True):
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
        top=float(np.max(bands) + 0.1 * np.max(bands)),
        bottom=float(np.min(bands) - 0.1 * np.abs(np.min(bands))),
    )
    ax.set_xticks(ticks, labels)

    ax = format_plot(ax)

    ax.set_ylabel(r"$E\ [t]$")

    return fig


def plot_superfluid_weight(
    x_data: npt.NDArray[np.float64],
    sf_weight_geom: npt.NDArray[np.float64],
    sf_weight_conv: npt.NDArray[np.float64],
    fig_in: matplotlib.figure.Figure | None = None,
    ax_in: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot superfluid weight against some parameter.

    Parameters
    ----------
    x_data : :class:`numpy.ndarray`
    sf_weight_geom : :class:`numpy.ndarray`
    sf_weight_conv : :class:`numpy.ndarray`
    fig_in : :class:`matplotlib.figure.Figure`, optional
    ax_in : :class:`matplotlib.axes.Axes`, optional

    Returns
    -------
    :obj:`matplotlib.figure.Figure`
        Figure with the data plotted onto the axis.

    """
    if fig_in is None or ax_in is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_in, ax_in

    ax.fill_between(
        x_data, 0, np.abs(sf_weight_geom), color="black", fc="#0271BB", label="geometric", hatch="-"
    )
    ax.fill_between(
        x_data,
        np.abs(sf_weight_geom),
        np.abs(sf_weight_geom) + np.abs(sf_weight_conv),
        color="black",
        fc="#E2001A",
        label="conventional",
        hatch="//",
    )
    ax.plot(x_data, np.abs(sf_weight_geom) + np.abs(sf_weight_conv), "x--", color="black")

    ax = format_plot(ax)
    ax.set_ylabel(r"$D_S\ [t]$")

    return fig
