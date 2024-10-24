# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Test the self-consistency loop."""

import numpy as np
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
            hubbard_int_gr=0,
            hubbard_int_x=0,
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
