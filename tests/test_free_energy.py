# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Tests for the free energy functions."""

import numpy as np
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture
from quant_met import mean_field, utils, geometry


def test_free_energy(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Test the free energy functions."""
    t_gr = 1
    t_x = 0.01
    v = 1
    chemical_potential = 1
    graphene_lattice = geometry.Graphene(lattice_constant=np.sqrt(3))
    bz_grid = graphene_lattice.generate_bz_grid(10, 10)

    egx_h = mean_field.EGXHamiltonian(
        hopping_gr=t_gr,
        hopping_x=t_x,
        hopping_x_gr_a=v,
        lattice_constant=graphene_lattice.lattice_constant,
        chemical_potential=chemical_potential,
        hubbard_int_gr=1,
        hubbard_int_x=1,
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
