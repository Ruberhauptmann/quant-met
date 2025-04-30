"""BdG Hamiltonian."""

import numpy as np
import numpy.typing as npt
import tbmodels


def bdg_hamiltonian(
    model: tbmodels.Model, k: npt.NDArray[np.floating]
) -> npt.NDArray[np.complexfloating]:
    """Generate the Bogoliubov-de Gennes (BdG) Hamiltonian.

    The BdG Hamiltonian incorporates pairing interactions and is used to
    study superfluid and superconducting phases. This method constructs a
    2x2 block Hamiltonian based on the normal state Hamiltonian and the
    pairing terms.

    Parameters
    ----------
    k : :class:`numpy.ndarray`
        List of k points in reciprocal space.

    Returns
    -------
    :class:`numpy.ndarray`
        The BdG Hamiltonian matrix evaluated at the specified k points.
    """
    if k.ndim == 1:
        k = np.expand_dims(k, axis=0)

    print(model, k)

    """
    h = np.zeros(
        (k.shape[0], 2 * self.number_of_bands, 2 * self.number_of_bands),
        dtype=np.complex128,
    )

    h[:, 0 : self.number_of_bands, 0 : self.number_of_bands] = self.hamiltonian(k)
    h[
        :,
        self.number_of_bands : 2 * self.number_of_bands,
        self.number_of_bands : 2 * self.number_of_bands,
    ] = -self.hamiltonian(self.q - k).conjugate()

    for i in range(self.number_of_bands):
        h[:, self.number_of_bands + i, i] = self.delta_orbital_basis[i]

    h[:, 0 : self.number_of_bands, self.number_of_bands : self.number_of_bands * 2] = (
        h[:, self.number_of_bands : self.number_of_bands * 2, 0 : self.number_of_bands]
        .copy()
        .conjugate()
    )

    return h.squeeze()
    """


def bdg_hamiltonian_derivative(
    model: tbmodels.Model, k: npt.NDArray[np.floating], direction: str
) -> npt.NDArray[np.complexfloating]:
    """Calculate the derivative of the BdG Hamiltonian.

    This method computes the spatial derivative of the Bogoliubov-de Gennes
    Hamiltonian with respect to the specified direction.

    Parameters
    ----------
    k : :class:`numpy.ndarray`
        List of k points in reciprocal space.
    direction : str
        Direction for the derivative, either 'x' or 'y'.

    Returns
    -------
    :class:`numpy.ndarray`
        The derivative of the BdG Hamiltonian matrix in the specified direction.
    """
    if k.ndim == 1:
        k = np.expand_dims(k, axis=0)

    print(k, model, direction)

    """
    h = np.zeros(
        (k.shape[0], 2 * self.number_of_bands, 2 * self.number_of_bands),
        dtype=np.complex128,
    )

    h[:, 0 : self.number_of_bands, 0 : self.number_of_bands] = self.hamiltonian_derivative(
        k, direction
    )
    h[
        :,
        self.number_of_bands : 2 * self.number_of_bands,
        self.number_of_bands : 2 * self.number_of_bands,
    ] = -self.hamiltonian_derivative(-k, direction).conjugate()

    return h.squeeze()
    """


def diagonalize_bdg(
    model: tbmodels.Model,
    k: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.complexfloating]]:
    """Diagonalizes the BdG Hamiltonian.

    This method computes the eigenvalues and eigenvectors of the Bogoliubov-de
    Gennes Hamiltonian, providing insight into the quasiparticle excitations in
    superconducting states.

    Parameters
    ----------
    k : :class:`numpy.ndarray`
        List of k points in reciprocal space.

    Returns
    -------
    tuple
        - :class:`numpy.ndarray`: Eigenvalues of the BdG Hamiltonian.
        - :class:`numpy.ndarray`: Eigenvectors corresponding to the eigenvalues of the
          BdG Hamiltonian.
    """
    print(model, k)
    """
    bdg_matrix = self.bdg_hamiltonian(k=k)
    if bdg_matrix.ndim == 2:
        bdg_matrix = np.expand_dims(bdg_matrix, axis=0)
        k = np.expand_dims(k, axis=0)

    bdg_wavefunctions = np.zeros(
        (len(k), 2 * self.number_of_bands, 2 * self.number_of_bands),
        dtype=np.complex128,
    )
    bdg_energies = np.zeros((len(k), 2 * self.number_of_bands))

    for i in range(len(k)):
        bdg_energies[i], bdg_wavefunctions[i] = np.linalg.eigh(bdg_matrix[i])

    return bdg_energies.squeeze(), bdg_wavefunctions.squeeze()
    """
