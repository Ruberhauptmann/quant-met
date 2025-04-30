"""Function to calculate superfluid weight."""

import numpy as np
import numpy.typing as npt
import tbmodels


def calculate_superfluid_weight(
    model: tbmodels.Model,
    k: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating]]:
    """Calculate the superfluid weight.

    Parameters
    ----------
    model : :class:`tbmodels.Model`
    Hamiltonian.
    k : :class:`numpy.ndarray`
    List of k points.

    Returns
    -------
    :class:`numpy.ndarray`
    Conventional contribution to the superfluid weight.
    :class:`numpy.ndarray`
    Geometric contribution to the superfluid weight.

    """
    print(model)
    s_weight_conv = np.zeros(shape=(2, 2), dtype=np.complex128)
    s_weight_geom = np.zeros(shape=(2, 2), dtype=np.complex128)

    c_mnpq_cache = {}

    for i, direction_1 in enumerate(["x", "y"]):
        for j, direction_2 in enumerate(["x", "y"]):
            for k_point in k:
                k_tuple = tuple(k_point)

                if k_tuple not in c_mnpq_cache:
                    c_mnpq_cache[k_tuple] = _c_factor(model, k_point)
                c_mnpq = c_mnpq_cache[k_tuple]

                j_up = _current_operator(model, direction_1, k_point)
                j_down = _current_operator(model, direction_2, -k_point)

                for m in range(model.size):
                    for n in range(model.size):
                        for p in range(model.size):
                            for q in range(model.size):
                                s_weight = c_mnpq[m, n, p, q] * j_up[m, n] * j_down[q, p]
                                if m == n and p == q:
                                    s_weight_conv[i, j] += s_weight
                                else:
                                    s_weight_geom[i, j] += s_weight

    return s_weight_conv, s_weight_geom


def _current_operator(
    model: tbmodels.Model, direction: str, k: npt.NDArray[np.floating]
) -> npt.NDArray[np.complexfloating]:
    j = np.zeros(shape=(model.size, model.size), dtype=np.complex128)
    print(direction)
    print(k)

    # _, bloch = self.diagonalize_nonint(k=k)

    """
    for m in range(self.number_of_bands):
        for n in range(self.number_of_bands):
            j[m, n] = (
                bloch[:, m].conjugate()
                @ self.hamiltonian_derivative(direction=direction, k=k)
                @ bloch[:, n]
            )
    """

    return j


def _c_factor(
    model: tbmodels.Model, k: npt.NDArray[np.floating]
) -> npt.NDArray[np.complexfloating]:
    # bdg_energies, bdg_functions = self.diagonalize_bdg(k)
    print(k)
    c_mnpq = np.zeros(
        shape=(
            model.size,
            model.size,
            model.size,
            model.size,
        ),
        dtype=np.complex128,
    )

    """
    for m in range(self.number_of_bands):
        for n in range(self.number_of_bands):
            for p in range(self.number_of_bands):
                for q in range(self.number_of_bands):
                    c_tmp: float = 0
                    for i in range(2 * self.number_of_bands):
                        for j in range(2 * self.number_of_bands):
                            if bdg_energies[i] != bdg_energies[j]:
                                c_tmp += (
                                    fermi_dirac(bdg_energies[i], self.beta)
                                    - fermi_dirac(bdg_energies[j], self.beta)
                                ) / (bdg_energies[i] - bdg_energies[j])
                            else:
                                c_tmp -= _fermi_dirac_derivative()

                            c_tmp *= (
                                bdg_functions[i, m].conjugate()
                                * bdg_functions[j, n]
                                * bdg_functions[j, p].conjugate()
                                * bdg_functions[i, q]
                            )

                    c_mnpq[m, n, p, q] = 2 * c_tmp
    """

    return c_mnpq


def _fermi_dirac_derivative() -> float:
    return 0


def _fermi_dirac() -> float:
    return 0
