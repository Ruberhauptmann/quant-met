import numpy as np
import numpy.typing as npt

from .configuration import Configuration


def generate_non_interacting_hamiltonian(k: npt.NDArray, config: Configuration):
    h = np.zeros((3, 3), dtype=complex)

    t_gr = config.t_gr
    t_x = config.t_x
    a = config.a
    V = config.V

    ## Double counting?
    # h[0, 1] = -0.5 * t_gr * (np.exp(1j * k[1] * a / np.sqrt(3)) + 2 * np.exp(-0.5j * a / np.sqrt(3) * k[1]) * (np.cos(0.5 * a * k[0])))
    h[0, 1] = t_gr * (
        np.exp(1j * k[1] * a / np.sqrt(3))
        + 2 * np.exp(-0.5j * a / np.sqrt(3) * k[1]) * (np.cos(0.5 * a * k[0]))
    )

    h[1, 0] = h[0, 1].conjugate()

    h[2, 1] = V
    h[1, 2] = V

    # h[2, 2] = - t_x * (
    #    np.cos(a * k[0]) +
    #    2 * np.cos(0.5 * a * k[0]) * np.cos(0.5 * np.sqrt(3) * a * k[1])
    # )
    h[2, 2] = (
        -2
        * t_x
        * (
            np.cos(a * k[0])
            + 2 * np.cos(0.5 * a * k[0]) * np.cos(0.5 * np.sqrt(3) * a * k[1])
        )
    )

    return h


def generate_bloch(k_points: npt.NDArray, config: Configuration):
    bloch = np.zeros((len(k_points), 3, 3), dtype=complex)
    energies = np.zeros((len(k_points), 3))

    for i, k in enumerate(k_points):
        h = generate_non_interacting_hamiltonian(k, config)

        energies[i], bloch[i] = np.linalg.eigh(h)
        energies[i] = energies[i] - config.mu

    bloch_absolute = np.power(np.absolute(bloch), 2)

    return energies, bloch_absolute
