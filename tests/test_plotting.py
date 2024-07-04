# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison

from quant_met import mean_field, plotting


@image_comparison(
    baseline_images=["scatter_into_bz"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_scatter_into_bz():
    lattice_constant = np.sqrt(3)

    all_K_points = (
        4
        * np.pi
        / (3 * lattice_constant)
        * np.array([(np.sin(i * np.pi / 6), np.cos(i * np.pi / 6)) for i in [1, 3, 5, 7, 9, 11]])
    )

    plotting.scatter_into_bz(bz_corners=all_K_points, k_points=np.array([[0, 0]]))


@image_comparison(
    baseline_images=["scatter_into_bz"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_scatter_into_bz_with_fig_in():
    lattice_constant = np.sqrt(3)

    all_K_points = (
        4
        * np.pi
        / (3 * lattice_constant)
        * np.array([(np.sin(i * np.pi / 6), np.cos(i * np.pi / 6)) for i in [1, 3, 5, 7, 9, 11]])
    )

    fig, ax = plt.subplots()

    plotting.scatter_into_bz(
        bz_corners=all_K_points, k_points=np.array([[0, 0]]), fig_in=fig, ax_in=ax
    )


@image_comparison(
    baseline_images=["scatter_into_bz_with_data"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_scatter_into_bz_with_data():
    lattice_constant = np.sqrt(3)

    all_K_points = (
        4
        * np.pi
        / (3 * lattice_constant)
        * np.array([(np.sin(i * np.pi / 6), np.cos(i * np.pi / 6)) for i in [1, 3, 5, 7, 9, 11]])
    )

    plotting.scatter_into_bz(
        bz_corners=all_K_points,
        k_points=np.array([[0, 0], [1, 1]]),
        data=np.array([1, 2]),
    )


@image_comparison(
    baseline_images=["nonint_bandstructure_graphene"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_plotting_nonint_bandstructure_graphene():
    lattice_constant = np.sqrt(3)
    all_K_points = (
        4
        * np.pi
        / (3 * lattice_constant)
        * np.array([(np.sin(i * np.pi / 6), np.cos(i * np.pi / 6)) for i in [1, 3, 5, 7, 9, 11]])
    )

    graphene_h = mean_field.GrapheneHamiltonian(t_nn=1, a=lattice_constant, mu=0, coulomb_gr=0)

    Gamma = np.array([0, 0])
    M = np.pi / lattice_constant * np.array([1, 1 / np.sqrt(3)])

    points = [(M, "M"), (Gamma, r"\Gamma"), (all_K_points[1], "K")]

    band_path, band_path_plot, ticks, labels = plotting.generate_bz_path(
        points, number_of_points=1000
    )

    band_structure = graphene_h.calculate_bandstructure(band_path)

    plotting.plot_bandstructure(
        bands=band_structure[["band_0", "band_1"]].to_numpy().T,
        k_point_list=band_path_plot,
        labels=labels,
        ticks=ticks,
    )


@image_comparison(
    baseline_images=["nonint_bandstructure_graphene"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_plotting_nonint_bandstructure_graphene_with_fig_in():
    lattice_constant = np.sqrt(3)
    all_K_points = (
        4
        * np.pi
        / (3 * lattice_constant)
        * np.array([(np.sin(i * np.pi / 6), np.cos(i * np.pi / 6)) for i in [1, 3, 5, 7, 9, 11]])
    )

    graphene_h = mean_field.GrapheneHamiltonian(t_nn=1, a=lattice_constant, mu=0, coulomb_gr=0)

    Gamma = np.array([0, 0])
    M = np.pi / lattice_constant * np.array([1, 1 / np.sqrt(3)])

    points = [(M, "M"), (Gamma, r"\Gamma"), (all_K_points[1], "K")]

    band_path, band_path_plot, ticks, labels = plotting.generate_bz_path(
        points, number_of_points=1000
    )

    band_structure = graphene_h.calculate_bandstructure(band_path)

    fig, ax = plt.subplots()

    plotting.plot_bandstructure(
        bands=band_structure[["band_0", "band_1"]].to_numpy().T,
        k_point_list=band_path_plot,
        labels=labels,
        ticks=ticks,
        fig_in=fig,
        ax_in=ax,
    )


@image_comparison(
    baseline_images=["nonint_bandstructure_egx"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_plotting_nonint_bandstructure_egx():
    lattice_constant = np.sqrt(3)
    all_K_points = (
        4
        * np.pi
        / (3 * lattice_constant)
        * np.array([(np.sin(i * np.pi / 6), np.cos(i * np.pi / 6)) for i in [1, 3, 5, 7, 9, 11]])
    )

    egx_h = mean_field.EGXHamiltonian(
        hopping_gr=1,
        hopping_x=0.01,
        hopping_x_gr_a=1,
        lattice_constant=lattice_constant,
        mu=0,
        coloumb_gr=0,
        coloumb_x=0,
    )

    Gamma = np.array([0, 0])
    M = np.pi / lattice_constant * np.array([1, 1 / np.sqrt(3)])

    points = [(M, "M"), (Gamma, r"\Gamma"), (all_K_points[1], "K")]

    band_path, band_path_plot, ticks, labels = plotting.generate_bz_path(
        points, number_of_points=1000
    )

    band_structure = egx_h.calculate_bandstructure(
        band_path, overlaps=(np.array([0, 0, 1]), np.array([1, 0, 0]))
    )

    plotting.plot_bandstructure(
        bands=band_structure[["band_0", "band_1", "band_2"]].to_numpy().T,
        overlaps=band_structure[["wx_0", "wx_1", "wx_2"]].to_numpy().T,
        k_point_list=band_path_plot,
        labels=labels,
        ticks=ticks,
    )


@image_comparison(
    baseline_images=["nonint_bandstructure_egx"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_plotting_nonint_bandstructure_egx_with_fig_in():
    lattice_constant = np.sqrt(3)
    all_K_points = (
        4
        * np.pi
        / (3 * lattice_constant)
        * np.array([(np.sin(i * np.pi / 6), np.cos(i * np.pi / 6)) for i in [1, 3, 5, 7, 9, 11]])
    )

    egx_h = mean_field.EGXHamiltonian(
        hopping_gr=1,
        hopping_x=0.01,
        hopping_x_gr_a=1,
        lattice_constant=lattice_constant,
        mu=0,
        coloumb_gr=0,
        coloumb_x=0,
    )

    Gamma = np.array([0, 0])
    M = np.pi / lattice_constant * np.array([1, 1 / np.sqrt(3)])

    points = [(M, "M"), (Gamma, r"\Gamma"), (all_K_points[1], "K")]

    band_path, band_path_plot, ticks, labels = plotting.generate_bz_path(
        points, number_of_points=1000
    )

    band_structure = egx_h.calculate_bandstructure(
        band_path, overlaps=(np.array([0, 0, 1]), np.array([1, 0, 0]))
    )

    fig, ax = plt.subplots()

    plotting.plot_bandstructure(
        bands=band_structure[["band_0", "band_1", "band_2"]].to_numpy().T,
        overlaps=band_structure[["wx_0", "wx_1", "wx_2"]].to_numpy().T,
        k_point_list=band_path_plot,
        labels=labels,
        ticks=ticks,
        fig_in=fig,
        ax_in=ax,
    )


def test_generate_bz_path():
    lattice_constant = np.sqrt(3)

    all_K_points = (
        4
        * np.pi
        / (3 * lattice_constant)
        * np.array([(np.sin(i * np.pi / 6), np.cos(i * np.pi / 6)) for i in [1, 3, 5, 7, 9, 11]])
    )
    Gamma = np.array([0, 0])
    M = np.pi / lattice_constant * np.array([1, 1 / np.sqrt(3)])

    points = [(M, "M"), (Gamma, r"\Gamma"), (all_K_points[1], "K")]
    band_path, band_path_plot, ticks, labels = plotting.generate_bz_path(
        points, number_of_points=1000
    )

    assert labels == ["$M$", "$\\Gamma$", "$K$", "$M$"]
    assert ticks[0] == 0.0
    assert band_path_plot[0] == 0.0
