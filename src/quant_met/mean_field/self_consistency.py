# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Self-consistency loop."""

import numpy as np
import numpy.typing as npt

from quant_met.mean_field.hamiltonians.base_hamiltonian import BaseHamiltonian


def self_consistency_loop(
    h: BaseHamiltonian,
    k_space_grid: npt.NDArray[np.float64],
    beta: np.float64,
    epsilon: float,
    q: npt.NDArray[np.float64] | None = None,
) -> BaseHamiltonian:
    """Self-consistency loop.

    Parameters
    ----------
    lattice
    q
    beta
    number_of_k_points
    h
    epsilon
    """
    if q is None:
        q = np.array([0, 0])

    rng = np.random.default_rng()
    delta_init = np.zeros(shape=h.delta_orbital_basis.shape, dtype=np.complex64)
    delta_init += (
        2 * rng.random(size=h.delta_orbital_basis.shape)
        - 1
        + 1.0j * (2 * rng.random(size=h.delta_orbital_basis.shape) - 1)
    )
    h.delta_orbital_basis = delta_init

    while True:
        new_gap = h.gap_equation(k=k_space_grid, q=q, beta=beta)
        if (np.abs(h.delta_orbital_basis - new_gap) < epsilon).all():
            h.delta_orbital_basis = new_gap
            return h
        mixing_greed = 0.1
        h.delta_orbital_basis = mixing_greed * new_gap + (1 - mixing_greed) * h.delta_orbital_basis
