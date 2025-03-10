# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Regression tests for mean_field."""

import numpy as np
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from quant_met import geometry, mean_field, parameters


def test_density_of_states(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for density of states."""
    hopping = 1
    chemical_potential = 0

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

    dos = graphene_h.calculate_density_of_states(k=bz_grid)

    ndarrays_regression.check(
        {
            "DOS": dos,
        },
        default_tolerance={"atol": 1e-8, "rtol": 1e-8},
    )


def test_free_energy(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for free energy."""
    hopping = 1
    chemical_potential = 0

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

    graphene_h.beta = 50
    free_energy_finite_temperature = graphene_h.calculate_free_energy(k=bz_grid)

    ndarrays_regression.check(
        {
            "free_energy_finite_temperature": free_energy_finite_temperature,
        },
        default_tolerance={"atol": 1e-8, "rtol": 1e-8},
    )


def test_current(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for current."""
    hopping = 1
    chemical_potential = 0

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

    graphene_h.beta = 50
    current = graphene_h.calculate_current_density(k=bz_grid)

    ndarrays_regression.check(
        {"current": current},
        default_tolerance={"atol": 1e-8, "rtol": 1e-8},
    )


def test_spectral_gap(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for calculation of spectral gap."""
    hopping = 1
    chemical_potential = 0

    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))
    bz_grid = graphene_lattice.generate_bz_grid(10, 10)
    dressed_graphene_h = mean_field.hamiltonians.DressedGraphene(
        parameters=parameters.DressedGrapheneParameters(
            hopping_gr=hopping,
            hopping_x=0.01,
            hopping_x_gr_a=1.0,
            lattice_constant=graphene_lattice.lattice_constant,
            chemical_potential=chemical_potential,
            hubbard_int_orbital_basis=[1.0, 1.0, 1.0],
            delta=np.array([0, 0, 0], dtype=np.complex128),
        )
    )
    spectral_gap_zero_gap = dressed_graphene_h.calculate_spectral_gap(k=bz_grid)

    dressed_graphene_h = mean_field.hamiltonians.DressedGraphene(
        parameters=parameters.DressedGrapheneParameters(
            hopping_gr=hopping,
            hopping_x=0.01,
            hopping_x_gr_a=1.0,
            lattice_constant=graphene_lattice.lattice_constant,
            chemical_potential=chemical_potential,
            hubbard_int_orbital_basis=[1.0, 1.0, 1.0],
            delta=np.array([1, 1, 1], dtype=np.complex128),
        )
    )
    spectral_gap_finite_gap = dressed_graphene_h.calculate_spectral_gap(k=bz_grid)

    ndarrays_regression.check(
        {
            "zero gap": spectral_gap_zero_gap,
            "finite gap": spectral_gap_finite_gap,
        },
        default_tolerance={"atol": 1e-8, "rtol": 1e-8},
    )


def test_hamiltonian_derivative_graphene(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for the derivative of the Graphene Hamiltonian."""
    hopping = 1
    chemical_potential = 0

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

    h_der_x = graphene_h.hamiltonian_derivative(k=bz_grid, direction="x")
    h_der_y = graphene_h.hamiltonian_derivative(k=bz_grid, direction="y")

    ndarrays_regression.check(
        {
            "h_der_x": h_der_x,
            "h_der_y": h_der_y,
        },
        default_tolerance={"atol": 1e-8, "rtol": 1e-8},
    )


def test_hamiltonian_derivative_one_band(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for the derivative of the one band Hamiltonian."""
    hopping = 1
    chemical_potential = 0

    square_lattice = geometry.SquareLattice(lattice_constant=1)
    bz_grid = square_lattice.generate_bz_grid(10, 10)
    one_band_h = mean_field.hamiltonians.OneBand(
        parameters=parameters.OneBandParameters(
            hopping=hopping,
            lattice_constant=square_lattice.lattice_constant,
            chemical_potential=chemical_potential,
            hubbard_int_orbital_basis=[1.0],
            delta=np.array([1], dtype=np.complex128),
        )
    )

    h_der_x = one_band_h.hamiltonian_derivative(k=bz_grid, direction="x")
    h_der_y = one_band_h.hamiltonian_derivative(k=bz_grid, direction="y")
    h_der_x_one_point = one_band_h.hamiltonian_derivative(k=np.array([1, 1]), direction="x")
    h_der_y_one_point = one_band_h.hamiltonian_derivative(k=np.array([1, 1]), direction="y")

    ndarrays_regression.check(
        {
            "h_der_x": h_der_x,
            "h_der_y": h_der_y,
            "h_der_x_one_point": h_der_x_one_point,
            "h_der_y_one_point": h_der_y_one_point,
        },
        default_tolerance={"atol": 1e-8, "rtol": 1e-8},
    )


def test_hamiltonian_two_band(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for the derivative of the two band Hamiltonian."""
    hopping = 1
    chemical_potential = 0

    square_lattice = geometry.SquareLattice(lattice_constant=1)
    bz_grid = square_lattice.generate_bz_grid(10, 10)
    two_band_h = mean_field.hamiltonians.TwoBand(
        parameters=parameters.TwoBandParameters(
            hopping=hopping,
            lattice_constant=square_lattice.lattice_constant,
            chemical_potential=chemical_potential,
            hubbard_int_orbital_basis=[1.0, 1.0],
            delta=np.array([1.0, 1.0], dtype=np.complex128),
        )
    )

    h = two_band_h.hamiltonian(k=bz_grid)
    h_one_point = two_band_h.hamiltonian(k=np.array([1, 1]))
    h_der_x = two_band_h.hamiltonian_derivative(k=bz_grid, direction="x")
    h_der_y = two_band_h.hamiltonian_derivative(k=bz_grid, direction="y")
    h_der_x_one_point = two_band_h.hamiltonian_derivative(k=np.array([1, 1]), direction="x")
    h_der_y_one_point = two_band_h.hamiltonian_derivative(k=np.array([1, 1]), direction="y")

    ndarrays_regression.check(
        {
            "h": h,
            "h_one_point": h_one_point,
            "h_der_x": h_der_x,
            "h_der_y": h_der_y,
            "h_der_x_one_point": h_der_x_one_point,
            "h_der_y_one_point": h_der_y_one_point,
        },
        default_tolerance={"atol": 1e-8, "rtol": 1e-8},
    )


def test_hamiltonian_three_band(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for the derivative of the two band Hamiltonian."""
    hopping = 1
    chemical_potential = 0

    square_lattice = geometry.SquareLattice(lattice_constant=1)
    bz_grid = square_lattice.generate_bz_grid(10, 10)
    three_band_h = mean_field.hamiltonians.ThreeBand(
        parameters=parameters.ThreeBandParameters(
            hopping=hopping,
            lattice_constant=square_lattice.lattice_constant,
            chemical_potential=chemical_potential,
            hubbard_int_orbital_basis=[1.0, 1.0, 1.0],
            delta=np.array([1.0, 1.0, 1.0], dtype=np.complex128),
        )
    )

    h = three_band_h.hamiltonian(k=bz_grid)
    h_one_point = three_band_h.hamiltonian(k=np.array([1, 1]))
    h_der_x = three_band_h.hamiltonian_derivative(k=bz_grid, direction="x")
    h_der_y = three_band_h.hamiltonian_derivative(k=bz_grid, direction="y")
    h_der_x_one_point = three_band_h.hamiltonian_derivative(k=np.array([1, 1]), direction="x")
    h_der_y_one_point = three_band_h.hamiltonian_derivative(k=np.array([1, 1]), direction="y")

    ndarrays_regression.check(
        {
            "h": h,
            "h_one_point": h_one_point,
            "h_der_x": h_der_x,
            "h_der_y": h_der_y,
            "h_der_x_one_point": h_der_x_one_point,
            "h_der_y_one_point": h_der_y_one_point,
        },
        default_tolerance={"atol": 1e-8, "rtol": 1e-8},
    )


def test_bdg_hamiltonian_derivative_graphene(
    ndarrays_regression: NDArraysRegressionFixture,
) -> None:
    """Test the derivative of the Graphene BdG Hamiltonian."""
    hopping = 1
    chemical_potential = 0

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

    h_der_x = graphene_h.bdg_hamiltonian_derivative(k=bz_grid, direction="x")
    h_der_y = graphene_h.bdg_hamiltonian_derivative(k=bz_grid, direction="y")
    h_der_x_one_point = graphene_h.bdg_hamiltonian_derivative(k=np.array([1, 1]), direction="x")
    h_der_y_one_point = graphene_h.bdg_hamiltonian_derivative(k=np.array([1, 1]), direction="y")

    ndarrays_regression.check(
        {
            "h_der_x": h_der_x,
            "h_der_y": h_der_y,
            "h_der_x_one_point": h_der_x_one_point,
            "h_der_y_one_point": h_der_y_one_point,
        },
        default_tolerance={"atol": 1e-8, "rtol": 1e-8},
    )
