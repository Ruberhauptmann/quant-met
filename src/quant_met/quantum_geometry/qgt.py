"""Calculate the quantum geometric tensor."""

import numpy as np
import numpy.typing as npt
import sisl


def calculate_qgt(
    hamiltonian: sisl.Hamiltonian,
    k_grid: npt.NDArray[np.floating],
    bands: list[int],
) -> npt.NDArray[np.floating]:
    """Calculate the quantum geometric tensor for selected bands."""
    qgt = np.zeros((2, 2), dtype=np.complex128)

    for k_point in k_grid:
        # Diagonalize at this k-point
        hk = hamiltonian.Hk(k=k_point, format="array")
        energies, bloch = np.linalg.eigh(hk)

        # Derivatives of H at this k-point
        der_h_k = hamiltonian.dHk(k=k_point, format="array")

        for band in bands:
            for i, der_h_i in enumerate(der_h_k):  # i: x=0, y=1
                for j, der_h_j in enumerate(der_h_k):  # j: x=0, y=1
                    for n in range(len(energies)):
                        if n == band:
                            continue
                        denom = (energies[band] - energies[n]) ** 2
                        if np.isclose(denom, 0.0):
                            continue  # avoid division by zero
                        mni = np.vdot(bloch[:, band], der_h_i @ bloch[:, n])
                        mnj = np.vdot(bloch[:, n], der_h_j @ bloch[:, band])
                        qgt[i, j] += mni * mnj / denom

    return np.real(qgt) / len(k_grid)
