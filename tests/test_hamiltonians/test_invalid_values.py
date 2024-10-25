# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Tests that invalid values are correctly identified."""

import numpy as np
import pytest
from pydantic import ValidationError
from quant_met import mean_field, parameters


def test_invalid_values_graphene() -> None:
    """Test that invalid values are correctly identified in graphene."""
    with pytest.raises(ValidationError, match=r"3 validation errors for GrapheneParameters.*"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=-1, lattice_constant=-1, chemical_potential=1, hubbard_int=-1
                )
            )
        )


def test_invalid_values_dressed_graphene() -> None:
    """Test that invalid values are correctly identified in dressed graphene."""
    with pytest.raises(
        ValidationError, match=r"4 validation errors for DressedGrapheneParameters.*"
    ):
        print(
            mean_field.hamiltonians.DressedGraphene(
                parameters=parameters.DressedGrapheneParameters(
                    hopping_gr=-1,
                    hopping_x=-1,
                    hopping_x_gr_a=-1,
                    lattice_constant=-1,
                    chemical_potential=1,
                    hubbard_int_orbital_basis=[-1.0, -1.0, -1.0],
                )
            )
        )


def test_invalid_values_one_band() -> None:
    """Test that invalid values are correctly identified in dressed graphene."""
    with pytest.raises(ValidationError, match=r"3 validation errors for OneBandParameters.*"):
        print(
            mean_field.hamiltonians.OneBand(
                parameters=parameters.OneBandParameters(
                    hopping=-1, lattice_constant=-1, chemical_potential=1, hubbard_int=-1
                )
            )
        )


def test_invalid_k_values() -> None:
    """Test that invalid k values are correctly identified."""
    with pytest.raises(ValueError, match="k is NaN or Infinity"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=1,
                    lattice_constant=1,
                    chemical_potential=1,
                    hubbard_int_orbital_basis=[1.0, 1.0],
                )
            ).hamiltonian(k=np.array([np.nan, np.nan]))
        )
    with pytest.raises(ValueError, match="k is NaN or Infinity"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=1,
                    lattice_constant=1,
                    chemical_potential=1,
                    hubbard_int_orbital_basis=[1.0, 1.0],
                )
            ).hamiltonian(k=np.array([[np.nan, np.inf]]))
        )
    with pytest.raises(ValueError, match="k is NaN or Infinity"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=1,
                    lattice_constant=1,
                    chemical_potential=1,
                    hubbard_int_orbital_basis=[1.0, 1.0],
                )
            ).bdg_hamiltonian(k=np.array([np.nan, np.nan]))
        )
    with pytest.raises(ValueError, match="k is NaN or Infinity"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=1,
                    lattice_constant=1,
                    chemical_potential=1,
                    hubbard_int_orbital_basis=[1.0, 1.0],
                )
            ).hamiltonian_derivative(k=np.array([np.nan, np.nan]), direction="x")
        )
