# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT
"""Test superfluid weight."""

import numpy as np
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture
from quant_met import mean_field, utils, geometry


def test_superfluid_weight_egx(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Test superfluid weight for EGX."""
    t_gr = 1
    t_x = 0.01
    v = 1
    mu = 1

    graphene_lattice = geometry.Graphene()
    bz_grid = graphene_lattice.generate_bz_grid(10, 10)

    egx_h = mean_field.EGXHamiltonian(
        hopping_gr=t_gr,
        hopping_x=t_x,
        hopping_x_gr_a=v,
        lattice_constant=graphene_lattice.lattice_constant,
        mu=mu,
        coloumb_gr=1,
        coloumb_x=1,
        delta=np.array([1, 1, 1]),
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

    graphene_lattice = geometry.Graphene()
    bz_grid = graphene_lattice.generate_bz_grid(10, 10)

    graphene_h = mean_field.GrapheneHamiltonian(
        t_nn=t_nn,
        a=graphene_lattice.lattice_constant,
        mu=mu,
        coulomb_gr=1,
        delta=np.array([1, 1]),
    )

    d_s_conv, d_s_geom = mean_field.superfluid_weight(h=graphene_h, k_grid=bz_grid)

    ndarrays_regression.check(
        {
            "D_S_conv": np.array(d_s_conv),
            "D_S_geom": np.array(d_s_geom),
        },
        default_tolerance={"atol": 1e-2, "rtol": 1e-4},
    )
