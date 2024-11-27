# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Self-consistency loop."""

import logging

import numpy as np
import numpy.typing as npt

from quant_met.mean_field.hamiltonians.base_hamiltonian import BaseHamiltonian
from quant_met.parameters import GenericParameters

logger = logging.getLogger(__name__)


def self_consistency_loop(
    h: BaseHamiltonian[GenericParameters],
    k_space_grid: npt.NDArray[np.float64],
    epsilon: float,
    max_iter: int = 1000,
) -> BaseHamiltonian[GenericParameters]:
    """Self-consistently solves the gap equation for a given Hamiltonian.

    This function performs a self-consistency loop to solve the gap equation
    for a Hamiltonian `h`.
    The gaps in the orbital basis are iteratively updated until the change is within
    a specified tolerance `epsilon`.

    Parameters
    ----------
    h : :class:`BaseHamiltonian<quant_met.mean_field.hamiltonians.BaseHamiltonian>`
        The Hamiltonian object with the parameters for the calculation.

    k_space_grid : :class:`numpy.ndarray`
        A grid of points in the Brillouin zone at which the gap equation is evaluated.
        See

    epsilon : float
        The convergence criterion. The loop will terminate when the change
        in the delta orbital basis is less than this value.

    max_iter : int
        Maximal number of iterations, default 300.

    Returns
    -------
    :class:`quant_met.mean_field.BaseHamiltonian`
        The updated Hamiltonian object with the new gaps.

    Notes
    -----
    The function initializes the gaps with random complex numbers before entering the
    self-consistency loop.
    The mixing parameter is set to 0.2, which controls how much of the new gaps is taken
    relative to the previous value in each iteration.
    """
    logger.info("Starting self-consistency loop.")

    rng = np.random.default_rng()
    delta_init = np.zeros(shape=h.delta_orbital_basis.shape, dtype=np.complex64)
    delta_init += (0.2 * rng.random(size=h.delta_orbital_basis.shape) - 1) + 1.0j * (
        0.2 * rng.random(size=h.delta_orbital_basis.shape) - 1
    )
    h.delta_orbital_basis = delta_init
    logger.debug("Initial gaps set to: %s", h.delta_orbital_basis)

    iteration_count = 0
    while True:
        iteration_count += 1
        if iteration_count > max_iter:
            msg = "Maximum number of iterations reached."
            raise RuntimeError(msg)

        logger.debug("Iteration %d: Computing new gaps.", iteration_count)

        new_gap = h.gap_equation(k=k_space_grid)

        logger.debug("New gaps computed: %s", new_gap)

        if np.allclose(h.delta_orbital_basis, new_gap, atol=1e-10, rtol=epsilon):
            h.delta_orbital_basis = new_gap
            logger.info("Convergence achieved after %d iterations.", iteration_count)
            return h

        mixing_greed = 0.2
        h.delta_orbital_basis = mixing_greed * new_gap + (1 - mixing_greed) * h.delta_orbital_basis
        logger.debug("Updated gaps: %s", h.delta_orbital_basis)
        logger.debug("Change in gaps: %s", np.abs(h.delta_orbital_basis - new_gap))
