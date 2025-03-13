# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

import numpy as np
from triqs.lattice.tight_binding import TBLattice

from quant_met.mean_field.hamiltonians import BaseHamiltonian
from quant_met.parameters import GenericParameters, HamiltonianParameters


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


def _tbl_factory(h: BaseHamiltonian[GenericParameters]) -> TBLattice:
    lattice_constant = np.sqrt(3)

    basis_vectors = [
        0.5 * lattice_constant * np.array([1, np.sqrt(3), 0]),
        0.5 * lattice_constant * np.array([1, -np.sqrt(3), 0]),
    ]
    orbital_positions = [
        (0.5 * (np.sqrt(3) - 1), 0, 0),
        (0.5 * (np.sqrt(3) + 1), 0, 0),
        (0.5 * (np.sqrt(3) - 1), 0, 0),
    ]
    hoppings = {
        (0, 0): [
            [0, h.hopping_gr, h.hopping_x_gr_a],
            [h.hopping_gr, 0, 0],
            [h.hopping_x_gr_a, 0, 0],
        ],
        (1, 0): [[0, 0, 0], [h.hopping_gr, 0, 0], [0, 0, 0]],
        (-1, 0): [[0, h.hopping_gr, 0], [0, 0, 0], [0, 0, 0]],
        (0, 1): [[0, h.hopping_gr, 0], [0, 0, 0], [0, 0, 0]],
        (0, -1): [[0, 0, 0], [h.hopping_gr, 0, 0], [0, 0, 0]],
    }

    return TBLattice(
        units=basis_vectors,
        hoppings=hoppings,
        orbital_positions=orbital_positions,
        orbital_names=["A", "B", "X"],
    )
