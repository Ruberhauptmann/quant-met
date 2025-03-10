# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Test Hamiltonian with fixed values."""

import numpy as np

from quant_met import geometry, mean_field, parameters


def test_hamiltonian_k_space_graphene() -> None:
    """Test Graphene Hamiltonians at some k points."""
    t_gr = 1
    chemical_potential = 1
    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))
    h_at_high_symmetry_points = [
        (
            graphene_lattice.Gamma,
            np.array(
                [[-chemical_potential, -3 * t_gr], [-3 * t_gr, -chemical_potential]],
                dtype=np.complex128,
            ),
        ),
        (
            graphene_lattice.K,
            np.array([[-chemical_potential, 0], [0, -chemical_potential]], dtype=np.complex128),
        ),
    ]

    for k_point, h_compare in h_at_high_symmetry_points:
        graphene_h = mean_field.hamiltonians.Graphene(
            parameters=parameters.GrapheneParameters(
                hopping=t_gr,
                lattice_constant=graphene_lattice.lattice_constant,
                chemical_potential=chemical_potential,
                hubbard_int_orbital_basis=[0.0, 0.0],
            )
        )
        h_generated = graphene_h.hamiltonian(k_point)
        assert np.allclose(h_generated, h_compare)


def test_hamiltonian_k_space_dressed_graphene() -> None:
    """Test dressed Graphene Hamiltonian at some k points."""
    t_gr = 1
    t_x = 0.01
    v = 1
    chemical_potential = 1

    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))
    h_at_high_symmetry_points = [
        (
            graphene_lattice.Gamma,
            np.array(
                [
                    [-chemical_potential, -3 * t_gr, v],
                    [-3 * t_gr, -chemical_potential, 0],
                    [v, 0, -chemical_potential - 6 * t_x],
                ],
                dtype=np.complex128,
            ),
        ),
    ]

    for k_point, h_compare in h_at_high_symmetry_points:
        dressed_graphene_h = mean_field.hamiltonians.DressedGraphene(
            parameters=parameters.DressedGrapheneParameters(
                hopping_gr=t_gr,
                hopping_x=t_x,
                hopping_x_gr_a=v,
                lattice_constant=graphene_lattice.lattice_constant,
                chemical_potential=chemical_potential,
                hubbard_int_orbital_basis=[0.0, 0.0, 0.0],
            )
        )
        h_generated = dressed_graphene_h.hamiltonian(k_point)
        assert np.allclose(h_generated, h_compare)


def test_hamiltonian_k_space_one_band() -> None:
    """Test one band Hamiltonian at some k points."""
    chemical_potential = 1

    square_lattice = geometry.SquareLattice(lattice_constant=1.0)
    h_at_high_symmetry_points = [
        (
            square_lattice.Gamma,
            np.array(
                [[-4 - chemical_potential]],
                dtype=np.complex128,
            ),
        ),
    ]

    for k_point, h_compare in h_at_high_symmetry_points:
        one_band_h = mean_field.hamiltonians.OneBand(
            parameters=parameters.OneBandParameters(
                hopping=1,
                lattice_constant=square_lattice.lattice_constant,
                chemical_potential=chemical_potential,
                hubbard_int_orbital_basis=[0.0],
            )
        )
        h_generated = one_band_h.hamiltonian(k_point)
        assert np.allclose(h_generated, h_compare)


def test_gap_equation_dressed_graphene_nonint() -> None:
    """Test gap equation for dressed Graphene model."""
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
    assert np.allclose(
        dressed_graphene_h.gap_equation(k=graphene_lattice.generate_bz_grid(ncols=30, nrows=30)),
        np.zeros(3, dtype=np.complex128),
    )
