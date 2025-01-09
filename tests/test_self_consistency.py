# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Test the self-consistency loop."""

import numpy as np
import pytest

from quant_met import geometry, mean_field, parameters


def test_self_consistency() -> None:
    """Test the self-consistency loop."""
    one_band_h = mean_field.hamiltonians.OneBand(
        parameters=parameters.OneBandParameters(
            hopping=1,
            lattice_constant=1,
            chemical_potential=0,
            hubbard_int_orbital_basis=[0.0],
        )
    )
    assert np.allclose(
        mean_field.self_consistency_loop(
            h=one_band_h,
            k_space_grid=one_band_h.lattice.generate_bz_grid(ncols=40, nrows=40),
            epsilon=1e-3,
        ).delta_orbital_basis,
        np.zeros(1),
    )


def test_self_consistency_max_iter() -> None:
    """Test that the self-consistency loop exits after ."""
    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))

    dressed_graphene_h = mean_field.hamiltonians.DressedGraphene(
        parameters=parameters.DressedGrapheneParameters(
            hopping_gr=1,
            hopping_x=0.01,
            hopping_x_gr_a=1,
            lattice_constant=graphene_lattice.lattice_constant,
            chemical_potential=0,
            hubbard_int_orbital_basis=[0.0, 0.0, 0.0],
        )
    )

    with pytest.raises(RuntimeError):
        mean_field.self_consistency_loop(
            h=dressed_graphene_h,
            k_space_grid=graphene_lattice.generate_bz_grid(40, 40),
            epsilon=1e-3,
            max_iter=3,
        )
    with pytest.raises(RuntimeError):
        mean_field.self_consistency_loop(
            h=dressed_graphene_h,
            k_space_grid=graphene_lattice.generate_bz_grid(40, 40),
            epsilon=1e-3,
            delta_init=np.array([1, 1, 1], dtype=np.complex128),
            max_iter=3,
        )
