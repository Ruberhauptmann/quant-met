import numpy as np
from quant_met import geometry

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
    band_path, band_path_plot, ticks, labels = geometry.generate_bz_path(
        points, number_of_points=1000
    )

    assert labels == ["$M$", "$\\Gamma$", "$K$", "$M$"]
    assert ticks[0] == 0.0
    assert band_path_plot[0] == 0.0
