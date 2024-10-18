# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Tests for calculating the quantum metric."""

import numpy as np
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture
from quant_met import mean_field, utils, geometry


def test_quantum_metric_egx(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for calculating the quantum metric."""
    t_gr = 1
    t_x = 0.01
    v = 1
    chemical_potential = 1

    graphene_lattice = geometry.Graphene(lattice_constant=np.sqrt(3))
    bz_grid = graphene_lattice.generate_bz_grid(20, 20)

    egx_h = mean_field.hamiltonians.EGXHamiltonian(
        hopping_gr=t_gr,
        hopping_x=t_x,
        hopping_x_gr_a=v,
        lattice_constant=graphene_lattice.lattice_constant,
        chemical_potential=chemical_potential,
        hubbard_int_gr=1,
        hubbard_int_x=1,
        delta=np.array([1, 1, 1]),
    )

    quantum_metric_0 = mean_field.quantum_metric(h=egx_h, k_grid=bz_grid, bands=[0])
    quantum_metric_1 = mean_field.quantum_metric(h=egx_h, k_grid=bz_grid, bands=[1])
    quantum_metric_2 = mean_field.quantum_metric(h=egx_h, k_grid=bz_grid, bands=[2])

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

    graphene_lattice = geometry.Graphene(lattice_constant=np.sqrt(3))
    bz_grid = graphene_lattice.generate_bz_grid(20, 20)

    graphene_h = mean_field.hamiltonians.GrapheneHamiltonian(
        hopping=hopping,
       lattice_constant=graphene_lattice.lattice_constant,
        chemical_potential=chemical_potential,
        hubbard_int_gr=1,
        delta=np.array([1, 1]),
    )

    quantum_metric_0 = mean_field.quantum_metric(h=graphene_h, k_grid=bz_grid, bands=[0])
    quantum_metric_1 = mean_field.quantum_metric(h=graphene_h, k_grid=bz_grid, bands=[1])

    ndarrays_regression.check(
        {
            "quantum_metric_0": quantum_metric_0,
            "quantum_metric_1": quantum_metric_1,
        },
    )


def test_quantum_metric_bdg_egx(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for calculating the quantum metric."""
    t_gr = 1
    t_x = 0.01
    v = 1
    chemical_potential = 1

    graphene_lattice = geometry.Graphene(lattice_constant=np.sqrt(3))
    bz_grid = graphene_lattice.generate_bz_grid(20, 20)

    egx_h = mean_field.hamiltonians.EGXHamiltonian(
        hopping_gr=t_gr,
        hopping_x=t_x,
        hopping_x_gr_a=v,
        lattice_constant=graphene_lattice.lattice_constant,
        chemical_potential=chemical_potential,
        hubbard_int_gr=1,
        hubbard_int_x=1,
        delta=np.array([1, 1, 1]),
    )

    quantum_metric_0 = mean_field.quantum_metric_bdg(h=egx_h, k_grid=bz_grid, bands=[0])
    quantum_metric_1 = mean_field.quantum_metric_bdg(h=egx_h, k_grid=bz_grid, bands=[1])
    quantum_metric_2 = mean_field.quantum_metric_bdg(h=egx_h, k_grid=bz_grid, bands=[2])

    ndarrays_regression.check(
        {
            "quantum_metric_0": quantum_metric_0,
            "quantum_metric_1": quantum_metric_1,
            "quantum_metric_2": quantum_metric_2,
        },
    )


def test_quantum_metric_bdg_graphene(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for calculating the quantum metric."""
    hopping = 1
    chemical_potential = 1

    graphene_lattice = geometry.Graphene(lattice_constant=np.sqrt(3))
    bz_grid = graphene_lattice.generate_bz_grid(20, 20)

    graphene_h = mean_field.hamiltonians.GrapheneHamiltonian(
        hopping=hopping,
       lattice_constant=graphene_lattice.lattice_constant,
        chemical_potential=chemical_potential,
        hubbard_int_gr=1,
        delta=np.array([1, 1]),
    )

    quantum_metric_0 = mean_field.quantum_metric_bdg(h=graphene_h, k_grid=bz_grid, bands=[0])
    quantum_metric_1 = mean_field.quantum_metric_bdg(h=graphene_h, k_grid=bz_grid, bands=[1])

    ndarrays_regression.check(
        {
            "quantum_metric_0": quantum_metric_0,
            "quantum_metric_1": quantum_metric_1,
        },
    )
