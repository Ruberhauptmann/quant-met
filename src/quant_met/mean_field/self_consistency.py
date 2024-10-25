# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Self-consistency loop."""

import numpy as np
import numpy.typing as npt

from quant_met.mean_field.hamiltonians.base_hamiltonian import BaseHamiltonian
from quant_met.parameters import GenericParameters


def self_consistency_loop(
    h: BaseHamiltonian[GenericParameters],
    k_space_grid: npt.NDArray[np.float64],
    epsilon: float,
) -> BaseHamiltonian[GenericParameters]:
    """Self-consistency loop.

    Parameters
    ----------
    k_space_grid
    h
    epsilon
    """
    rng = np.random.default_rng()
    delta_init = np.zeros(shape=h.delta_orbital_basis.shape, dtype=np.complex64)
    delta_init += (
        2 * rng.random(size=h.delta_orbital_basis.shape)
        - 1
        + 1.0j * (2 * rng.random(size=h.delta_orbital_basis.shape) - 1)
    )
    h.delta_orbital_basis = delta_init

    while True:
        new_gap = h.gap_equation(k=k_space_grid)
        if (np.abs(h.delta_orbital_basis - new_gap) < epsilon).all():
            h.delta_orbital_basis = new_gap
            return h
        mixing_greed = 0.1
        h.delta_orbital_basis = mixing_greed * new_gap + (1 - mixing_greed) * h.delta_orbital_basis
