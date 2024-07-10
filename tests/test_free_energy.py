# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Tests for the free energy functions."""

import numpy as np
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture
from quant_met import mean_field, utils


def test_free_energy(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Test the free energy functions."""
    t_gr = 1
    t_x = 0.01
    v = 1
    mu = 1
    lattice_constant = np.sqrt(3)
    bz_corner_points = (
        4
        * np.pi
        / (3 * lattice_constant)
        * np.array([(np.sin(i * np.pi / 6), np.cos(i * np.pi / 6)) for i in [1, 3, 5, 7, 9, 11]])
    )
    egx_h = mean_field.EGXHamiltonian(
        hopping_gr=t_gr,
        hopping_x=t_x,
        hopping_x_gr_a=v,
        lattice_constant=lattice_constant,
        mu=mu,
        coloumb_gr=1,
        coloumb_x=1,
    )

    bz_grid = utils.generate_uniform_grid(
        10, 10, bz_corner_points[1], bz_corner_points[5], origin=np.array([0, 0])
    )

    delta_list = np.array([np.array([i, i, i]) for i in np.linspace(0, 4, 10)])

    free_energy_real_gap = np.array(
        [
            mean_field.free_energy_real_gap(delta_vector=delta, hamiltonian=egx_h, k_points=bz_grid)
            for delta in delta_list
        ]
    )
    free_energy_uniform_pairing = np.array(
        [
            mean_field.free_energy_uniform_pairing(
                delta=delta[0], hamiltonian=egx_h, k_points=bz_grid
            )
            for delta in delta_list
        ]
    )
    delta_list_complex = np.array([np.array([i, i, i, i, i, i]) for i in np.linspace(0, 4, 10)])
    free_energy_complex_gap = np.array(
        [
            mean_field.free_energy_complex_gap(
                delta_vector=delta, hamiltonian=egx_h, k_points=bz_grid
            )
            for delta in delta_list_complex
        ]
    )
    ndarrays_regression.check(
        {
            "delta_list": delta_list,
            "free_energy_real_gap": free_energy_real_gap,
            "free_energy_complex_gap": free_energy_complex_gap,
            "free_energy_uniform_pairing": free_energy_uniform_pairing,
        },
        default_tolerance={"atol": 1e-8, "rtol": 1e-8},
    )
