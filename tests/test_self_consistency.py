# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Test the self-consistency loop."""

import numpy as np
import pytest
from quant_met import geometry, mean_field, parameters


def test_self_consistency() -> None:
    """Test the self-consistency loop."""
    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))

    egx_h = mean_field.hamiltonians.DressedGraphene(
        parameters=parameters.DressedGrapheneParameters(
            hopping_gr=1,
            hopping_x=0.01,
            hopping_x_gr_a=1,
            lattice_constant=graphene_lattice.lattice_constant,
            chemical_potential=0,
            hubbard_int_orbital_basis=[0.0, 0.0, 0.0],
        )
    )
    assert np.allclose(
        mean_field.self_consistency_loop(
            h=egx_h, k_space_grid=graphene_lattice.generate_bz_grid(50, 50), epsilon=1e-4
        ).delta_orbital_basis,
        np.zeros(3),
    )
    assert np.allclose(
        mean_field.self_consistency_loop(
            h=egx_h,
            k_space_grid=graphene_lattice.generate_bz_grid(50, 50),
            epsilon=1e-4,
        ).delta_orbital_basis,
        np.zeros(3),
    )


def test_self_consistency_max_iter() -> None:
    """Test that the self-consistency loop exits after ."""
    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))

    egx_h = mean_field.hamiltonians.DressedGraphene(
        parameters=parameters.DressedGrapheneParameters(
            hopping_gr=1,
            hopping_x=0.01,
            hopping_x_gr_a=1,
            lattice_constant=graphene_lattice.lattice_constant,
            chemical_potential=0,
            hubbard_int_orbital_basis=[0.0, 0.0, 0.0],
        )
    )

    with pytest.raises(SystemExit):
        mean_field.self_consistency_loop(
            h=egx_h,
            k_space_grid=graphene_lattice.generate_bz_grid(50, 50),
            epsilon=1e-4,
            max_iter=3,
        )
