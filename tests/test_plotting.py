import numpy as np
from matplotlib.testing.decorators import image_comparison

from quant_met import hamiltonians, plotting


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
        * np.array(
            [
                (np.sin(i * np.pi / 6), np.cos(i * np.pi / 6))
                for i in [1, 3, 5, 7, 9, 11]
            ]
        )
    )

    plotting.scatter_into_bz(bz_corners=all_K_points, k_points=np.array([[0, 0]]))


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
        * np.array(
            [
                (np.sin(i * np.pi / 6), np.cos(i * np.pi / 6))
                for i in [1, 3, 5, 7, 9, 11]
            ]
        )
    )

    graphene_h = hamiltonians.GrapheneHamiltonian(
        t_nn=1, a=lattice_constant, mu=0, coulomb_gr=0
    )

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
        * np.array(
            [
                (np.sin(i * np.pi / 6), np.cos(i * np.pi / 6))
                for i in [1, 3, 5, 7, 9, 11]
            ]
        )
    )

    egx_h = hamiltonians.EGXHamiltonian(
        t_gr=1, t_x=0.01, V=1, a=lattice_constant, mu=0, U_gr=0, U_x=0
    )

    Gamma = np.array([0, 0])
    M = np.pi / lattice_constant * np.array([1, 1 / np.sqrt(3)])

    points = [(M, "M"), (Gamma, r"\Gamma"), (all_K_points[1], "K")]

    band_path, band_path_plot, ticks, labels = plotting.generate_bz_path(
        points, number_of_points=1000
    )

    band_structure = egx_h.calculate_bandstructure(band_path)

    plotting.plot_bandstructure(
        bands=band_structure[["band_0", "band_1", "band_2"]].to_numpy().T,
        overlaps=band_structure[["wx_0", "wx_1", "wx_2"]].to_numpy().T,
        k_point_list=band_path_plot,
        labels=labels,
        ticks=ticks,
    )


def test_generate_bz_path():
    lattice_constant = np.sqrt(3)

    all_K_points = (
        4
        * np.pi
        / (3 * lattice_constant)
        * np.array(
            [
                (np.sin(i * np.pi / 6), np.cos(i * np.pi / 6))
                for i in [1, 3, 5, 7, 9, 11]
            ]
        )
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