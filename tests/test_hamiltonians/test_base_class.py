# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Test for the Hamiltonian base class."""

import numpy as np
import pytest
from quant_met import mean_field
from quant_met.parameters.hamiltonians import HamiltonianParameters


@pytest.fixture()
def _patch_abstract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the abstract methods."""
    monkeypatch.setattr(mean_field.hamiltonians.BaseHamiltonian, "__abstractmethods__", set())


@pytest.mark.usefixtures("_patch_abstract")
def test_base_hamiltonian() -> None:
    """Test that the methods in the BaseHamiltonian class raises correct errors."""
    test_parameters = HamiltonianParameters(
        name="Test", beta=100.0, q=np.array([1, 1]), hubbard_int_orbital_basis=[1.0]
    )
    with pytest.raises(NotImplementedError):
        print(mean_field.hamiltonians.BaseHamiltonian(parameters=test_parameters))
