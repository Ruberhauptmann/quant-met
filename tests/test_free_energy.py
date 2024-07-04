# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

import numpy as np

from quant_met import mean_field, utils


def test_free_energy(ndarrays_regression):
    t_gr = 1
    t_x = 0.01
    V = 1
    mu = 1
    lattice_constant = np.sqrt(3)
    all_K_points = (
        4
        * np.pi
        / (3 * lattice_constant)
        * np.array([(np.sin(i * np.pi / 6), np.cos(i * np.pi / 6)) for i in [1, 3, 5, 7, 9, 11]])
    )
    egx_h = mean_field.EGXHamiltonian(
        hopping_gr=t_gr,
        hopping_x=t_x,
        hopping_x_gr_a=V,
        lattice_constant=lattice_constant,
        mu=mu,
        coloumb_gr=1,
        coloumb_x=1,
    )

    BZ_grid = utils.generate_uniform_grid(
        10, 10, all_K_points[1], all_K_points[5], origin=np.array([0, 0])
    )

    delta_list = np.array([np.array([i, i, i]) for i in np.linspace(0, 4, 10)])

    free_energy = np.array(
        [
            mean_field.free_energy(delta_vector=delta, hamiltonian=egx_h, k_points=BZ_grid)
            for delta in delta_list
        ]
    )
    free_energy_uniform_pairing = np.array(
        [
            mean_field.free_energy_uniform_pairing(
                delta=delta[0], hamiltonian=egx_h, k_points=BZ_grid
            )
            for delta in delta_list
        ]
    )

    ndarrays_regression.check(
        {
            "delta_list": delta_list,
            "free_energy": free_energy,
            "free_energy_uniform_pairing": free_energy_uniform_pairing,
        },
        default_tolerance=dict(atol=1e-8, rtol=1e-8),
    )
