"""Gap equation."""

import numpy as np
import numpy.typing as npt
import sisl
from numba import jit

from .bdg_hamiltonian import diagonalize_bdg


def gap_equation(
    hamiltonian: sisl.Hamiltonian,
    k: npt.NDArray[np.float64],
    beta: float,
    hubbard_int_orbital_basis: npt.NDArray[np.float64],
    delta_orbital_basis: npt.NDArray[np.float64],
) -> npt.NDArray[np.complexfloating]:
    """Gap equation.

    Parameters
    ----------
    delta_orbital_basis
    hubbard_int_orbital_basis
    beta
    hamiltonian
    k : :class:`numpy.ndarray`
        k grid

    Returns
    -------
    New delta
    """
    bdg_energies, bdg_wavefunctions = diagonalize_bdg(
        hamiltonian=hamiltonian, k=k, delta_orbital_basis=delta_orbital_basis
    )
    delta = np.zeros(hamiltonian.no, dtype=np.complex128)
    return gap_equation_loop(
        bdg_energies, bdg_wavefunctions, delta, beta, hubbard_int_orbital_basis, k
    )


@jit
def gap_equation_loop(
    bdg_energies: npt.NDArray[np.float64],
    bdg_wavefunctions: npt.NDArray[np.complex128],
    delta: npt.NDArray[np.complex128],
    beta: float,
    hubbard_int_orbital_basis: npt.NDArray[np.float64],
    k: npt.NDArray[np.floating],
) -> npt.NDArray[np.complexfloating]:
    """Calculate the gap equation.

    The gap equation determines the order parameter for superconductivity by
    relating the pairings to the spectral properties of the BdG Hamiltonian.

    Parameters
    ----------
    bdg_energies : :class:`numpy.ndarray`
        BdG energies
    bdg_wavefunctions : :class:`numpy.ndarray`
        BdG wavefunctions
    delta : :class:`numpy.ndarray`
        Delta
    beta : :class:`float`
        Beta
    hubbard_int_orbital_basis : :class:`numpy.ndarray`
        Hubard interaction in orbital basis
    k : :class:`numpy.ndarray`
        List of k points in reciprocal space.

    Returns
    -------
    :class:`numpy.ndarray`
        New pairing gap in orbital basis, adjusted to remove global phase.
    """
    number_of_bands = len(delta)
    for i in range(number_of_bands):
        sum_tmp = 0
        for j in range(2 * number_of_bands):
            for k_index in range(len(k)):
                sum_tmp += (
                    np.conjugate(bdg_wavefunctions[k_index, i, j])
                    * bdg_wavefunctions[k_index, i + number_of_bands, j]
                    * fermi_dirac(bdg_energies[k_index, j].item(), beta)
                )
        delta[i] = (-hubbard_int_orbital_basis[i] * sum_tmp / len(k)).conjugate()

    delta_without_phase: npt.NDArray[np.complexfloating] = delta * np.exp(
        -1j * np.angle(delta[np.argmax(np.abs(delta))])
    )
    return delta_without_phase


@jit
def fermi_dirac(energy: npt.NDArray[np.floating], beta: float) -> npt.NDArray[np.floating]:
    """Fermi dirac distribution.

    Parameters
    ----------
    energy
    beta

    Returns
    -------
    fermi_dirac

    """
    return (
        np.where(energy < 0, 1.0, 0.0)
        if np.isinf(beta)
        else np.asarray(1 / (1 + np.exp(beta * energy)))
    )
