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
    with pytest.raises(ValidationError, match=r"4 validation errors for GrapheneParameters.*"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=np.inf,
                    lattice_constant=np.inf,
                    chemical_potential=np.inf,
                    hubbard_int=np.inf,
                )
            )
        )
    with pytest.raises(ValidationError, match=r"4 validation errors for GrapheneParameters.*"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=np.nan,
                    lattice_constant=np.nan,
                    chemical_potential=np.nan,
                    hubbard_int=np.nan,
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
