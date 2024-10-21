# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Functions to run self-consistent calculation for the order parameter."""

from pydantic import BaseModel

from quant_met.mean_field.hamiltonians import BaseHamiltonian
from quant_met.parameters import Parameters


def _hamiltonian_factory(classname: str, parameters: BaseModel) -> BaseHamiltonian:
    """Create a hamiltonian by its class name."""
    from quant_met.mean_field import hamiltonians

    cls = getattr(hamiltonians, classname)
    h: BaseHamiltonian = cls(parameters)
    return h


def scf(parameters: Parameters) -> None:
    """Self-consistent calculation for the order parameter."""
    h = _hamiltonian_factory(parameters=parameters.model, classname=parameters.model.name)
    print(h.delta_orbital_basis)
