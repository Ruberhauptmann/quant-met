"""Calculate the supercurrent density."""

import numpy as np
import numpy.typing as npt
import tbmodels


def calculate_current_density(
    model: tbmodels.Model, k: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Calculate the current density.

    Parameters
    ----------
    model : tbmodels.Model
    k

    Returns
    -------
    current_density

    """
    print(model)

    # bdg_energies, bdg_wavefunctions = self.diagonalize_bdg(k=k)
    # h_der_x = self.hamiltonian_derivative(k=k, direction="x")
    # h_der_y = self.hamiltonian_derivative(k=k, direction="y")

    current = np.zeros(2, dtype=np.complex128)

    """
    matrix_x = np.zeros((3, 3), dtype=np.complex128)
    matrix_y = np.zeros((3, 3), dtype=np.complex128)
    for k_index in range(len(k)):
        for i in range(self.number_of_bands):
            for j in range(self.number_of_bands):
                for n in range(2 * self.number_of_bands):
                    matrix_x[i, j] += (
                        h_der_x[k_index, i, j]
                        * np.conjugate(bdg_wavefunctions[k_index, i, n])
                        * bdg_wavefunctions[k_index, j, n]
                        * _fermi_dirac(bdg_energies[k_index, n].item(), self.beta)
                    )
                    matrix_y[i, j] += (
                        h_der_y[k_index, i, j]
                        * np.conjugate(bdg_wavefunctions[k_index, i, n])
                        * bdg_wavefunctions[k_index, j, n]
                        * _fermi_dirac(bdg_energies[k_index, n].item(), self.beta)
                    )

    current[0] = np.sum(matrix_x, axis=None)
    current[1] = np.sum(matrix_y, axis=None)
    assert np.allclose(np.imag(current), 0, atol=1e-12)
    """

    return (2 * np.real(current)) / len(k)


def _fermi_dirac() -> float:
    return 0
