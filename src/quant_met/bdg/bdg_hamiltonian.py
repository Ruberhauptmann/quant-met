"""BdG Hamiltonian."""

import numpy as np
import numpy.typing as npt
import sisl


def bdg_hamiltonian(
    hamiltonian: sisl.Hamiltonian,
    k: npt.NDArray[np.floating],
    delta_orbital_basis: npt.NDArray[np.complexfloating],
    q: npt.NDArray[np.floating] = None,
) -> npt.NDArray[np.complexfloating]:
    """
    Construct the BdG Hamiltonian at momentum k.

    Parameters
    ----------
    hamiltonian : sisl.Hamiltonian
        The normal-state tight-binding Hamiltonian.
    k : np.ndarray
        k-point(s) in reduced coordinates. Shape: (3,) or (N_k, 3).
    delta_orbital_basis : np.ndarray
        Pairing amplitudes in the orbital basis. Shape: (N_orbitals,)
    q : np.ndarray, optional
        Pairing momentum (e.g. for FFLO). Default is 0.

    Returns
    -------
    np.ndarray
        The BdG Hamiltonian. Shape: (2N, 2N) or (N_k, 2N, 2N)
    """
    k = np.atleast_2d(k)
    n_k_points = k.shape[0]
    n_orbitals = hamiltonian.no

    if q is None:
        q = np.zeros(3)

    h_bdg = np.zeros((n_k_points, 2 * n_orbitals, 2 * n_orbitals), dtype=np.complex128)

    for i, kpt in enumerate(k):
        h_k = hamiltonian.Hk(kpt).toarray()
        h_mkq = hamiltonian.Hk(q - kpt).toarray()

        h_bdg[i, :n_orbitals, :n_orbitals] = h_k
        h_bdg[i, n_orbitals:, n_orbitals:] = -h_mkq.conj()

        for j in range(n_orbitals):
            h_bdg[i, n_orbitals + j, j] = delta_orbital_basis[j]

        h_bdg[i, :n_orbitals, n_orbitals:] = h_bdg[i, n_orbitals:, :n_orbitals].conj().T

    return h_bdg.squeeze()


def diagonalize_bdg(
    hamiltonian: sisl.Hamiltonian,
    kgrid: sisl.MonkhorstPack,
    delta_orbital_basis: np.ndarray,
    q: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """Diagonalizes the BdG Hamiltonian.

    This method computes the eigenvalues and eigenvectors of the Bogoliubov-de
    Gennes Hamiltonian, providing insight into the quasiparticle excitations in
    superconducting states.

    Parameters
    ----------
    q
    kgrid
    delta_orbital_basis
    hamiltonian

    Returns
    -------
    tuple
        - :class:`numpy.ndarray`: Eigenvalues of the BdG Hamiltonian.
        - :class:`numpy.ndarray`: Eigenvectors corresponding to the eigenvalues of the
          BdG Hamiltonian.
    """
    energies = []
    wavefunctions = []

    for kpt in kgrid:
        bdg = bdg_hamiltonian(
            hamiltonian=hamiltonian, delta_orbital_basis=delta_orbital_basis, k=kpt, q=q
        )
        e, v = np.linalg.eigh(bdg)
        energies.append(e)
        wavefunctions.append(v)

    return np.array(energies), np.array(wavefunctions)
