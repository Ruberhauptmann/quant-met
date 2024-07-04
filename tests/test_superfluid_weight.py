# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT
"""Test superfluid weight."""

import numpy as np
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture
from quant_met import mean_field, utils


def test_superfluid_weight_egx(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Test superfluid weight for EGX."""
    t_gr = 1
    t_x = 0.01
    v = 1
    mu = 1
    lattice_constant = np.sqrt(3)
    bz_corners = (
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
        delta=np.array([1, 1, 1]),
    )

    bz_grid = utils.generate_uniform_grid(
        10, 10, bz_corners[1], bz_corners[5], origin=np.array([0, 0])
    )

    d_s_conv, d_s_geom = mean_field.superfluid_weight(h=egx_h, k_grid=bz_grid)

    ndarrays_regression.check(
        {
            "D_S_conv": np.array(d_s_conv),
            "D_S_geom": np.array(d_s_geom),
        },
        default_tolerance={"atol": 1e-4, "rtol": 1e-4},
    )


def test_superfluid_weight_graphene(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Test superfluid weight for graphene."""
    t_nn = 1
    mu = 1
    lattice_constant = np.sqrt(3)
    bz_corners = (
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

    bz_grid = utils.generate_uniform_grid(
        10, 10, bz_corners[1], bz_corners[5], origin=np.array([0, 0])
    )

    d_s_conv, d_s_geom = mean_field.superfluid_weight(h=graphene_h, k_grid=bz_grid)

    ndarrays_regression.check(
        {
            "D_S_conv": np.array(d_s_conv),
            "D_S_geom": np.array(d_s_geom),
        },
        default_tolerance={"atol": 1e-2, "rtol": 1e-4},
    )
