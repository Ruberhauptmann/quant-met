# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Test superfluid weight."""

import numpy as np
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from quant_met import geometry, mean_field, parameters


def test_superfluid_weight_dressed_graphene(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Test superfluid weight for dressed_graphene."""
    t_gr = 1
    t_x = 0.01
    v = 1
    chemical_potential = 1

    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))
    bz_grid = graphene_lattice.generate_bz_grid(10, 10)

    dressed_graphene_h = mean_field.hamiltonians.DressedGraphene(
        parameters=parameters.DressedGrapheneParameters(
            hopping_gr=t_gr,
            hopping_x=t_x,
            hopping_x_gr_a=v,
            lattice_constant=graphene_lattice.lattice_constant,
            chemical_potential=chemical_potential,
            hubbard_int_orbital_basis=[1.0, 1.0, 1.0],
            delta=np.array([1, 1, 1], dtype=np.complex128),
        )
    )

    d_s_conv, d_s_geom = dressed_graphene_h.calculate_superfluid_weight(k=bz_grid)

    ndarrays_regression.check(
        {
            "D_S_conv": np.array(d_s_conv),
            "D_S_geom": np.array(d_s_geom),
        },
        default_tolerance={"atol": 0, "rtol": 1e-8},
    )


def test_superfluid_weight_graphene(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Test superfluid weight for graphene."""
    hopping = 1
    chemical_potential = 1

    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))
    bz_grid = graphene_lattice.generate_bz_grid(10, 10)

    graphene_h = mean_field.hamiltonians.Graphene(
        parameters=parameters.GrapheneParameters(
            hopping=hopping,
            lattice_constant=graphene_lattice.lattice_constant,
            chemical_potential=chemical_potential,
            hubbard_int_orbital_basis=[1.0, 1.0],
            delta=np.array([1, 1], dtype=np.complex128),
        )
    )

    d_s_conv, d_s_geom = graphene_h.calculate_superfluid_weight(k=bz_grid)

    ndarrays_regression.check(
        {
            "D_S_conv": np.array(d_s_conv),
            "D_S_geom": np.array(d_s_geom),
        },
        default_tolerance={"atol": 0, "rtol": 1e-8},
    )
