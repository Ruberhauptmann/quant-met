# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

from quant_met.mean_field.hamiltonians import BaseHamiltonian
from quant_met.parameters import HamiltonianParameters


def _hamiltonian_factory(
    classname: str, parameters: HamiltonianParameters
) -> BaseHamiltonian[HamiltonianParameters]:
    """Create a Hamiltonian by its class name.

    Parameters
    ----------
    classname: str
        The name of the Hamiltonian class to instantiate.
    parameters: HamiltonianParameters
        An instance of HamiltonianParameters containing all necessary
        configuration for the specific Hamiltonian.

    Returns
    -------
    BaseHamiltonian[HamiltonianParameters]
        An instance of the specified Hamiltonian class.
    """
    from quant_met.mean_field import hamiltonians

    cls = getattr(hamiltonians, classname)
    h: BaseHamiltonian[HamiltonianParameters] = cls(parameters)
    return h
