# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Tests for calculating the quantum metric."""

import numpy as np
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture
from quant_met import mean_field, utils


def test_quantum_metric_egx(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for calculating the quantum metric."""
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
        delta=np.array([1, 1, 1]),
    )

    bz_grid = utils.generate_uniform_grid(
        20, 20, bz_corner_points[1], bz_corner_points[5], origin=np.array([0, 0])
    )

    quantum_metric_0 = mean_field.quantum_metric(h=egx_h, k_grid=bz_grid, band=0)
    quantum_metric_1 = mean_field.quantum_metric(h=egx_h, k_grid=bz_grid, band=1)
    quantum_metric_2 = mean_field.quantum_metric(h=egx_h, k_grid=bz_grid, band=2)

    ndarrays_regression.check(
        {
            "quantum_metric_0": quantum_metric_0,
            "quantum_metric_1": quantum_metric_1,
            "quantum_metric_2": quantum_metric_2,
        },
    )


def test_quantum_metric_graphene(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for calculating the quantum metric."""
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
        20, 20, bz_corners[1], bz_corners[5], origin=np.array([0, 0])
    )

    quantum_metric_0 = mean_field.quantum_metric(h=graphene_h, k_grid=bz_grid, band=0)
    quantum_metric_1 = mean_field.quantum_metric(h=graphene_h, k_grid=bz_grid, band=1)

    ndarrays_regression.check(
        {
            "quantum_metric_0": quantum_metric_0,
            "quantum_metric_1": quantum_metric_1,
        },
    )
