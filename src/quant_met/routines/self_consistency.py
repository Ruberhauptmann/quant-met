"""Self-consistency loop."""

import logging

import numpy as np
import numpy.typing as npt
import sisl
from numpy import complexfloating, dtype, ndarray

from quant_met.bdg import gap_equation

logger = logging.getLogger(__name__)


def self_consistency_loop(
    hamiltonian: sisl.Hamiltonian,
    k_space_grid: npt.NDArray[np.floating],
    beta: float,
    hubbard_int_orbital_basis: npt.NDArray[np.float64],
    epsilon: float,
    max_iter: int = 1000,
    delta_init: npt.NDArray[np.complex128] | None = None,
) -> ndarray[tuple[int, ...], dtype[complexfloating]]:
    """Self-consistently solves the gap equation for a given Hamiltonian.

    This function performs a self-consistency loop to solve the gap equation
    for a Hamiltonian `h`.
    The gaps in the orbital basis are iteratively updated until the change is within
    a specified tolerance `epsilon`.

    Parameters
    ----------
    hubbard_int_orbital_basis
    beta
    hamiltonian : sisl.Hamiltonian
        The Hamiltonian object.

    k_space_grid : :class:`numpy.ndarray`
        A grid of points in the Brillouin zone at which the gap equation is evaluated.

    epsilon : float
        The convergence criterion. The loop will terminate when the change
        in the delta orbital basis is less than this value.

    delta_init : :class:`numpy.ndarray`
        Initial gaps in orbital basis.

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

    if delta_init is None:
        rng = np.random.default_rng()
        delta_init = np.zeros(shape=hamiltonian.no, dtype=np.complex128)
        delta_init += (0.2 * rng.random(size=hamiltonian.no) - 1) + 1.0j * (
            0.2 * rng.random(size=hamiltonian.no) - 1
        )
    logger.debug("Initial gaps set to: %s", delta_init)
    delta = delta_init

    iteration_count = 0
    while True:
        iteration_count += 1
        if iteration_count > max_iter:
            msg = "Maximum number of iterations reached."
            raise RuntimeError(msg)

        logger.debug("Iteration %d: Computing new gaps.", iteration_count)

        new_gap = gap_equation(
            hamiltonian=hamiltonian,
            k=k_space_grid,
            beta=beta,
            hubbard_int_orbital_basis=hubbard_int_orbital_basis,
            delta_orbital_basis=delta,
        )

        logger.debug("New gaps computed: %s", new_gap)

        if np.allclose(delta, new_gap, atol=1e-10, rtol=epsilon):
            logger.info("Convergence achieved after %d iterations.", iteration_count)
            return new_gap

        mixing_greed = 0.2
        delta = mixing_greed * new_gap + (1 - mixing_greed) * delta
