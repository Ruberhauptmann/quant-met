# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Functions to calculate the free energy of a BdG Hamiltonian."""

import numpy as np
import numpy.typing as npt

from quant_met.mean_field.hamiltonians import BaseHamiltonian
from quant_met.parameters import GenericParameters


def free_energy(
    hamiltonian: BaseHamiltonian[GenericParameters],
    k_points: npt.NDArray[np.float64],
) -> float:
    """Calculate the free energy of a BdG Hamiltonian.

    Parameters
    ----------
    hamiltonian : :class:`BaseHamiltonian`
        Hamiltonian to be evaluated.
    k_points : :class:`numpy.ndarray`
        List of k points

    Returns
    -------
    float
        Free energy from the BdG Hamiltonian.

    """
    number_k_points = len(k_points)
    bdg_energies, _ = hamiltonian.diagonalize_bdg(k_points)

    print(bdg_energies[0])

    k_array = np.array(
        [
            np.sum(np.log(1 + np.exp(-hamiltonian.beta * bdg_energies[k_index])))
            for k_index in range(number_k_points)
        ]
    )

    integral: float = -np.sum(k_array, axis=-1) / (hamiltonian.beta / number_k_points) - np.sum(
        np.power(np.abs(hamiltonian.delta_orbital_basis), 2) / hamiltonian.hubbard_int_orbital_basis
    )

    return integral
