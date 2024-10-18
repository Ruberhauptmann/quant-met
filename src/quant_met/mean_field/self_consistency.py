# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Self-consistency loop."""

import numpy as np
import numpy.typing as npt

from quant_met import geometry

from .base_hamiltonian import BaseHamiltonian


def self_consistency_loop(
    h: BaseHamiltonian,
    beta: np.float64,
    number_of_k_points: int,
    epsilon: float,
    q: npt.NDArray[np.float64] | None = None,
) -> BaseHamiltonian:
    """Self-consistency loop.

    Parameters
    ----------
    q
    beta
    number_of_k_points
    h
    epsilon
    """
    lattice = geometry.Graphene(lattice_constant=np.sqrt(3))
    k_space_grid = lattice.generate_bz_grid(ncols=number_of_k_points, nrows=number_of_k_points)
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
