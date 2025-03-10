# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Tests for calculating the quantum metric."""

import numpy as np
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from quant_met import geometry, mean_field, parameters


def test_quantum_metric_dressed_graphene(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for calculating the quantum metric."""
    t_gr = 1
    t_x = 0.01
    v = 1
    chemical_potential = 1

    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))
    bz_grid = graphene_lattice.generate_bz_grid(20, 20)

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

    quantum_metric_0 = dressed_graphene_h.calculate_quantum_metric(k=bz_grid, bands=[0])
    quantum_metric_1 = dressed_graphene_h.calculate_quantum_metric(k=bz_grid, bands=[1])
    quantum_metric_2 = dressed_graphene_h.calculate_quantum_metric(k=bz_grid, bands=[2])

    ndarrays_regression.check(
        {
            "quantum_metric_0": quantum_metric_0,
            "quantum_metric_1": quantum_metric_1,
            "quantum_metric_2": quantum_metric_2,
        },
    )


def test_quantum_metric_graphene(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for calculating the quantum metric."""
    hopping = 1
    chemical_potential = 1

    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))
    bz_grid = graphene_lattice.generate_bz_grid(20, 20)

    graphene_h = mean_field.hamiltonians.Graphene(
        parameters=parameters.GrapheneParameters(
            hopping=hopping,
            lattice_constant=graphene_lattice.lattice_constant,
            chemical_potential=chemical_potential,
            hubbard_int_orbital_basis=[1.0, 1.0],
            delta=np.array([1, 1], dtype=np.complex128),
        )
    )

    quantum_metric_0 = graphene_h.calculate_quantum_metric(k=bz_grid, bands=[0])
    quantum_metric_1 = graphene_h.calculate_quantum_metric(k=bz_grid, bands=[1])

    ndarrays_regression.check(
        {
            "quantum_metric_0": quantum_metric_0,
            "quantum_metric_1": quantum_metric_1,
        },
    )
