# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison

from quant_met import mean_field, plotting, geometry


@image_comparison(
    baseline_images=["scatter_into_bz"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_scatter_into_bz():
    graphene_lattice = geometry.Graphene(lattice_constant=np.sqrt(3))
    plotting.scatter_into_bz(bz_corners=graphene_lattice.bz_corners, k_points=np.array([[0, 0]]))


@image_comparison(
    baseline_images=["scatter_into_bz"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_scatter_into_bz_with_fig_in():
    graphene_lattice = geometry.Graphene(lattice_constant=np.sqrt(3))
    fig, ax = plt.subplots()
    plotting.scatter_into_bz(
        bz_corners=graphene_lattice.bz_corners, k_points=np.array([[0, 0]]), fig_in=fig, ax_in=ax
    )


@image_comparison(
    baseline_images=["scatter_into_bz_with_data"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_scatter_into_bz_with_data():
    graphene_lattice = geometry.Graphene(lattice_constant=np.sqrt(3))
    plotting.scatter_into_bz(
        bz_corners=graphene_lattice.bz_corners,
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
    graphene_lattice = geometry.Graphene(lattice_constant=np.sqrt(3))
    graphene_h = mean_field.GrapheneHamiltonian(hopping=1,lattice_constant=graphene_lattice.lattice_constant, chemical_potential=0, hubbard_int_gr=0)

    points = [(graphene_lattice.M, "M"), (graphene_lattice.Gamma, r"\Gamma"), (graphene_lattice.K, "K")]
    band_path, band_path_plot, ticks, labels = geometry.generate_bz_path(
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
    graphene_lattice = geometry.Graphene(lattice_constant=np.sqrt(3))

    graphene_h = mean_field.GrapheneHamiltonian(hopping=1,lattice_constant=graphene_lattice.lattice_constant, chemical_potential=0, hubbard_int_gr=0)
    points = [(graphene_lattice.M, "M"), (graphene_lattice.Gamma, r"\Gamma"), (graphene_lattice.K, "K")]

    band_path, band_path_plot, ticks, labels = geometry.generate_bz_path(
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
    graphene_lattice = geometry.Graphene(lattice_constant=np.sqrt(3))
    egx_h = mean_field.EGXHamiltonian(
        hopping_gr=1,
        hopping_x=0.01,
        hopping_x_gr_a=1,
        lattice_constant=graphene_lattice.lattice_constant,
        chemical_potential=0,
        hubbard_int_gr=0,
        hubbard_int_x=0,
    )
    points = [(graphene_lattice.M, "M"), (graphene_lattice.Gamma, r"\Gamma"), (graphene_lattice.K, "K")]
    band_path, band_path_plot, ticks, labels = geometry.generate_bz_path(
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
    graphene_lattice = geometry.Graphene()
    egx_h = mean_field.EGXHamiltonian(
        hopping_gr=1,
        hopping_x=0.01,
        hopping_x_gr_a=1,
        lattice_constant=graphene_lattice.lattice_constant,
        chemical_potential=0,
        hubbard_int_gr=0,
        hubbard_int_x=0,
    )

    points = [(graphene_lattice.M, "M"), (graphene_lattice.Gamma, r"\Gamma"), (graphene_lattice.K, "K")]

    band_path, band_path_plot, ticks, labels = geometry.generate_bz_path(
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


@image_comparison(
    baseline_images=["nonint_bandstructure_egx"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_plotting_nonint_bandstructure_egx_with_fig_in():
    graphene_lattice = geometry.Graphene(lattice_constant=np.sqrt(3))
    egx_h = mean_field.EGXHamiltonian(
        hopping_gr=1,
        hopping_x=0.01,
        hopping_x_gr_a=1,
        lattice_constant=graphene_lattice.lattice_constant,
        chemical_potential=0,
        hubbard_int_gr=0,
        hubbard_int_x=0,
    )

    points = [(graphene_lattice.M, "M"), (graphene_lattice.Gamma, r"\Gamma"), (graphene_lattice.K, "K")]

    band_path, band_path_plot, ticks, labels = geometry.generate_bz_path(
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


@image_comparison(
    baseline_images=["sf_weight"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_plotting_sf_weight():
    plotting.plot_superfluid_weight(
        x_data=np.array([0.5, 1, 1.5, 2, 2.5, 3]),
        sf_weight_geom=np.array([1, 2, 3, 4, 5, 6]),
        sf_weight_conv=np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    )


@image_comparison(
    baseline_images=["sf_weight"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_plotting_sf_weight_with_fig_in():
    fig, ax = plt.subplots()
    plotting.plot_superfluid_weight(
        x_data=np.array([0.5, 1, 1.5, 2, 2.5, 3]),
        sf_weight_geom=np.array([1, 2, 3, 4, 5, 6]),
        sf_weight_conv=np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        fig_in=fig,
        ax_in=ax,
    )


@image_comparison(
    baseline_images=["quantum_metric"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_plotting_quantum_metric():
    plotting.plot_quantum_metric(
        x_data=np.array([0.5, 1, 1.5, 2, 2.5, 3]),
        quantum_metric=np.array([1, 2, 3, 4, 5, 6]),
    )


@image_comparison(
    baseline_images=["quantum_metric"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_plotting_quantum_metric_with_fig_in():
    fig, ax = plt.subplots()
    plotting.plot_quantum_metric(
        x_data=np.array([0.5, 1, 1.5, 2, 2.5, 3]),
        quantum_metric=np.array([1, 2, 3, 4, 5, 6]),
        fig_in=fig,
        ax_in=ax,
    )
