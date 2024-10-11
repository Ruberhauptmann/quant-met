# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Self-consistency loop."""

import numpy as np

from quant_met import geometry

from .base_hamiltonian import BaseHamiltonian


def self_consistency_loop(
    h: BaseHamiltonian, beta: np.float64, number_of_k_points: int, epsilon: float
) -> BaseHamiltonian:
    """Self-consistency loop.

    Parameters
    ----------
    beta
    number_of_k_points
    h
    epsilon
    """
    lattice = geometry.Graphene(lattice_constant=np.sqrt(3))
    k_space_grid = lattice.generate_bz_grid(ncols=number_of_k_points, nrows=number_of_k_points)
    rng = np.random.default_rng()
    delta_init = np.zeros(shape=h.delta_orbital_basis.shape, dtype=np.float64)
    rng.random(size=h.delta_orbital_basis.shape, out=delta_init)
    h.delta_orbital_basis = delta_init.astype(np.complex64)

    while True:
        new_gap = h.gap_equation(k=k_space_grid, beta=beta)
        if (np.abs(h.delta_orbital_basis - new_gap) < epsilon).all():
            h.delta_orbital_basis = new_gap
            return h
        mixing_greed = 0.2
        h.delta_orbital_basis = mixing_greed * new_gap + (1 - mixing_greed) * h.delta_orbital_basis
