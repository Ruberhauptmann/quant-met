# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Functions to calculate the free energy of a BdG Hamiltonian."""

import numpy as np
import numpy.typing as npt

from .base_hamiltonian import BaseHamiltonian


def free_energy(
    hamiltonian: BaseHamiltonian,
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

    k_array = np.array(
        [
            np.sum(np.abs(bdg_energies[k_index][0 : hamiltonian.number_of_bands]))
            for k_index in range(number_k_points)
        ]
    )

    integral: float = -np.sum(k_array, axis=-1) / number_k_points + np.sum(
        np.power(np.abs(hamiltonian.delta_orbital_basis), 2) / hamiltonian.coloumb_orbital_basis
    )

    return integral


def free_energy_complex_gap(
    delta_vector: npt.NDArray[np.float64],
    hamiltonian: BaseHamiltonian,
    k_points: npt.NDArray[np.float64],
) -> float:
    """Calculate the free energy of a BdG Hamiltonian, with a complex order parameter.

    Parameters
    ----------
    delta_vector : :class:`numpy.ndarray`
        Delta in orbital basis, with consecutive floats getting converted into one complex number,
        so [a, b, c, d] -> [a+b*j, c+d*j].
    hamiltonian : :class:`BaseHamiltonian`
        Hamiltonian to be evaluated.
    k_points : :class:`numpy.ndarray`
        List of k points

    Returns
    -------
    float
        Free energy from the BdG Hamiltonian.

    """
    hamiltonian.delta_orbital_basis = delta_vector[0::2] + 1j * delta_vector[1::2]

    return free_energy(hamiltonian=hamiltonian, k_points=k_points)


def free_energy_real_gap(
    delta_vector: npt.NDArray[np.float64],
    hamiltonian: BaseHamiltonian,
    k_points: npt.NDArray[np.float64],
) -> float:
    """Calculate the free energy of a BdG Hamiltonian, with a real order parameter.

    Parameters
    ----------
    delta_vector : :class:`numpy.ndarray`
        Delta in orbital basis.
    hamiltonian : :class:`BaseHamiltonian`
        Hamiltonian to be evaluated.
    k_points : :class:`numpy.ndarray`
        List of k points

    Returns
    -------
    float
        Free energy from the BdG Hamiltonian.

    """
    hamiltonian.delta_orbital_basis = delta_vector.astype(np.complex64)

    return free_energy(hamiltonian=hamiltonian, k_points=k_points)


def free_energy_uniform_pairing(
    delta: float,
    hamiltonian: BaseHamiltonian,
    k_points: npt.NDArray[np.float64],
) -> float:
    """Calculate the free energy of a BdG Hamiltonian, with uniform pairing constraint.

    Parameters
    ----------
    delta : float
        Delta.
    hamiltonian : :class:`BaseHamiltonian`
        Hamiltonian to be evaluated.
    k_points : :class:`numpy.ndarray`
        List of k points

    Returns
    -------
    float
        Free energy from the BdG Hamiltonian.

    """
    hamiltonian.delta_orbital_basis = np.full(
        hamiltonian.number_of_bands, fill_value=delta, dtype=np.float64
    ).astype(np.complex64)

    return free_energy(hamiltonian=hamiltonian, k_points=k_points)
