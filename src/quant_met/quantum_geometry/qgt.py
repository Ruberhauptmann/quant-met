"""Calculate the quantum geometric tensor."""

import numpy as np
import numpy.typing as npt
import tbmodels


def calculate_qgt(
    model: tbmodels.Model, k: npt.NDArray[np.floating], bands: list[int]
) -> npt.NDArray[np.floating]:
    """Calculate the quantum geometric tensor.

    This function computes the quantum geometric tensor associated with
    the specified bands of a given Hamiltonian over a grid of k-points.
    The output is a 2x2 matrix representing the quantum metric.
    It gets summed over the band specified in the list `bands`.

    Parameters
    ----------
    model : :class:`tbmodels.Model`
        Model for which the quantum geometric tensor is calculated.
    k : numpy.ndarray
        Array of k points in the Brillouin zone.
    bands : list of int
        Indices of the bands for which the quantum geometric tensor is calculated.

    Returns
    -------
    :class:`numpy.ndarray`
        A 2x2 matrix representing the quantum geometric tensor.

    Raises
    ------
    ValueError
        If `bands` contains invalid indices or `k_grid` is empty.
    """
    energies, bloch = (1, 1)
    print(model)

    number_k_points = len(k)

    quantum_geom_tensor = np.zeros(shape=(2, 2), dtype=np.complex128)

    for band in bands:
        for i, direction_1 in enumerate(["x", "y"]):
            # h_derivative_direction_1 = self.hamiltonian_derivative(k=k, direction=direction_1)
            h_derivative_direction_1 = direction_1
            for j, direction_2 in enumerate(["x", "y"]):
                # h_derivative_direction_2 = self.hamiltonian_derivative(k=k, direction=direction_2)
                h_derivative_direction_2 = direction_2
                for k_index in range(len(k)):
                    # for n in [m for m in range(self.number_of_bands) if m != band]:
                    for n in [m for m in range(3) if m != band]:
                        quantum_geom_tensor[i, j] += (
                            (
                                bloch[k_index][:, band].conjugate()
                                @ h_derivative_direction_1[k_index]
                                @ bloch[k_index][:, n]
                            )
                            * (
                                bloch[k_index][:, n].conjugate()
                                @ h_derivative_direction_2[k_index]
                                @ bloch[k_index][:, band]
                            )
                            / (energies[k_index][band] - energies[k_index][n]) ** 2
                        )

    return np.real(quantum_geom_tensor) / number_k_points
