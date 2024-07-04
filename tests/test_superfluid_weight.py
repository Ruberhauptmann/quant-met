# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

import numpy as np

from quant_met import mean_field, utils


def test_superfluid_weight_egx(ndarrays_regression):
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
        delta=np.array([1, 1, 1]),
    )

    BZ_grid = utils.generate_uniform_grid(
        20, 20, all_K_points[1], all_K_points[5], origin=np.array([0, 0])
    )

    D_S_xx = mean_field.superfluid_weight(h=egx_h, k_grid=BZ_grid, direction_1="x", direction_2="x")
    D_S_xy = mean_field.superfluid_weight(h=egx_h, k_grid=BZ_grid, direction_1="x", direction_2="y")
    D_S_yy = mean_field.superfluid_weight(h=egx_h, k_grid=BZ_grid, direction_1="y", direction_2="y")

    ndarrays_regression.check(
        {
            "D_S_xx": np.array(D_S_xx),
            "D_S_xy": np.array(D_S_xy),
            "D_S_yy": np.array(D_S_yy),
        },
        default_tolerance=dict(atol=1e-4, rtol=1e-4),
    )


def test_superfluid_weight_graphene(ndarrays_regression):
    t_nn = 1
    mu = 1
    lattice_constant = np.sqrt(3)
    all_K_points = (
        4
        * np.pi
        / (3 * lattice_constant)
        * np.array([(np.sin(i * np.pi / 6), np.cos(i * np.pi / 6)) for i in [1, 3, 5, 7, 9, 11]])
    )
    graphene_h = mean_field.GrapheneHamiltonian(
        t_nn=t_nn,
        a=lattice_constant,
        mu=mu,
        coulomb_gr=1,
        delta=np.array([1, 1]),
    )

    BZ_grid = utils.generate_uniform_grid(
        20, 20, all_K_points[1], all_K_points[5], origin=np.array([0, 0])
    )

    D_S_xx = mean_field.superfluid_weight(
        h=graphene_h, k_grid=BZ_grid, direction_1="x", direction_2="x"
    )
    D_S_xy = mean_field.superfluid_weight(
        h=graphene_h, k_grid=BZ_grid, direction_1="x", direction_2="y"
    )
    D_S_yy = mean_field.superfluid_weight(
        h=graphene_h, k_grid=BZ_grid, direction_1="y", direction_2="y"
    )

    ndarrays_regression.check(
        {
            "D_S_xx": np.array(D_S_xx),
            "D_S_xy": np.array(D_S_xy),
            "D_S_yy": np.array(D_S_yy),
        },
        default_tolerance=dict(atol=1e-3, rtol=1e-4),
    )
