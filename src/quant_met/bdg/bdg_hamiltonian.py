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
    Construct the BdG Hamiltonian at momentum k using sisl.

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
    if k.ndim == 1:
        k = np.expand_dims(k, axis=0)

    n_k_points = k.shape[0]
    n_orbitals = hamiltonian.no

    if q is None:
        q = np.zeros(3)

    h = np.zeros((n_k_points, 2 * n_orbitals, 2 * n_orbitals), dtype=np.complex128)

    for i, kpt in enumerate(k):
        h[i, 0:n_orbitals, 0:n_orbitals] = hamiltonian.Hk(kpt).toarray()
        h[i, n_orbitals : 2 * n_orbitals, n_orbitals : 2 * n_orbitals] = (
            -hamiltonian.Hk(q - kpt).toarray().conj()
        )

        for j in range(n_orbitals):
            h[i, n_orbitals + j, j] = delta_orbital_basis[j]

        h[i, 0:n_orbitals, n_orbitals : 2 * n_orbitals] = (
            h[i, n_orbitals : 2 * n_orbitals, 0:n_orbitals].conj().T
        )

    return h.squeeze()


def diagonalize_bdg(
    hamiltonian: sisl.Hamiltonian,
    k: npt.NDArray[np.floating],
    delta_orbital_basis: np.ndarray,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.complexfloating]]:
    """Diagonalizes the BdG Hamiltonian.

    This method computes the eigenvalues and eigenvectors of the Bogoliubov-de
    Gennes Hamiltonian, providing insight into the quasiparticle excitations in
    superconducting states.

    Parameters
    ----------
    delta_orbital_basis
    hamiltonian
    k : :class:`numpy.ndarray`
        List of k points in reciprocal space.

    Returns
    -------
    tuple
        - :class:`numpy.ndarray`: Eigenvalues of the BdG Hamiltonian.
        - :class:`numpy.ndarray`: Eigenvectors corresponding to the eigenvalues of the
          BdG Hamiltonian.
    """
    bdg_matrix = bdg_hamiltonian(
        hamiltonian=hamiltonian, k=k, delta_orbital_basis=delta_orbital_basis
    )

    if bdg_matrix.ndim == 2:
        bdg_matrix = np.expand_dims(bdg_matrix, axis=0)

    results = [np.linalg.eigh(np.array(bdg_matrix[i])) for i in range(len(bdg_matrix))]
    energies, wavefunctions = zip(*results, strict=False)

    return np.squeeze(np.array(energies)), np.squeeze(np.array(wavefunctions))
