"""Calculate supercurrent."""

import numpy as np
import sisl
from numpy.typing import NDArray


def calculate_current_density(
    hamiltonian: sisl.Hamiltonian,
    k: sisl.MonkhorstPack,
    bdg_energies: NDArray[np.floating],
    bdg_wavefunctions: NDArray[np.complexfloating],
    beta: float,
) -> NDArray[np.floating]:
    """Calculate current density from BdG wavefunctions and normal-state Hamiltonian derivatives.

    Parameters
    ----------
    hamiltonian : sisl.Hamiltonian
        The normal-state Hamiltonian.
    k : np.ndarray
        Array of k-points in the Brillouin zone.
    bdg_energies : np.ndarray
        BdG eigenvalues for each k-point.
    bdg_wavefunctions : np.ndarray
        BdG eigenvectors for each k-point.
    beta : float
        Inverse temperature (1 / k_B T).

    Returns
    -------
    np.ndarray
        Real current density vector (2D).
    """

    def fermi_dirac(e: float, beta: float) -> float:
        return 1.0 / (np.exp(beta * e) + 1)

    num_bands = hamiltonian.no
    current = np.zeros(2, dtype=np.complex128)

    for dir_idx, _direction in enumerate(["x", "y"]):
        matrix = np.zeros((num_bands, num_bands), dtype=np.complex128)

        for k_index, kpt in enumerate(k):
            dhk = hamiltonian.dHk(kpt, format="array")[dir_idx]

            for i in range(num_bands):
                for j in range(num_bands):
                    for n in range(2 * num_bands):
                        matrix[i, j] += (
                            dhk[i, j]
                            * np.conj(bdg_wavefunctions[k_index, i, n])
                            * bdg_wavefunctions[k_index, j, n]
                            * fermi_dirac(bdg_energies[k_index, n], beta)
                        )

        current[dir_idx] = np.sum(matrix)

    return (2 * np.real(current)) / len(k)
