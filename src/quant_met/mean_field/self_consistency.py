# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Self-consistency loop."""

import numpy as np

from quant_met import geometry

from .base_hamiltonian import BaseHamiltonian


def self_consistency_loop(
    h: BaseHamiltonian, number_of_k_points: int, epsilon: float
) -> BaseHamiltonian:
    """Self-consistency loop.

    Parameters
    ----------
    number_of_k_points
    h
    epsilon
    """
    lattice = geometry.Graphene()
    k_space_grid = lattice.generate_bz_grid(ncols=number_of_k_points, nrows=number_of_k_points)
    rng = np.random.default_rng()
    delta_init = np.zeros(shape=h.delta_orbital_basis.shape, dtype=np.float64)
    rng.random(size=h.delta_orbital_basis.shape, out=delta_init)
    h.delta_orbital_basis = delta_init.astype(np.complex64)

    while True:
        new_gap = h.gap_equation(k=k_space_grid)
        if (np.abs(h.delta_orbital_basis) - np.abs(new_gap) < epsilon).all():
            h.delta_orbital_basis = new_gap
            print("Finished")
            return h
        print(f"Old: {h.delta_orbital_basis}")
        print(f"New: {new_gap}")
        print(f"Difference {np.abs(h.delta_orbital_basis) - np.abs(new_gap)}")
        h.delta_orbital_basis = new_gap
