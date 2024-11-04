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
    """Self-consistency loop for updating the delta orbital basis.

    This function performs a self-consistency loop to solve the gap equation
    for a given Hamiltonian.
    The delta orbital basis is iteratively updated until the change is within
    a specified tolerance (epsilon).
    The updates are performed using a mixing approach to ensure stability in
    convergence.

    Parameters
    ----------
    h : class `BaseHamiltonian`
        The Hamiltonian object containing the current delta orbital basis
        and the method to compute the gap equation.

    k_space_grid : class `numpy.ndarray`
        A grid of k-space points at which the gap equation is evaluated.
        This defines the momentum space for the calculation.

    epsilon : float
        The convergence criterion. The loop will terminate when the change
        in the delta orbital basis is less than this value.

    Returns
    -------
    BaseHamiltonian[GenericParameters]
        The updated Hamiltonian object with the new delta orbital basis.

    Examples
    --------
    >>> h = BaseHamiltonian(parameters)
    >>> k_space = np.linspace(-np.pi, np.pi, num=100)
    >>> epsilon = 1e-5
    >>> updated_h = self_consistency_loop(h, k_space, epsilon)

    Notes
    -----
    The function initializes the delta orbital basis with random complex
    numbers before entering the self-consistency loop.
    The mixing parameter is set to 0.2, which controls how much of the
    new gap is taken relative to the previous value in each iteration.
    """
    rng = np.random.default_rng()
    delta_init = np.zeros(shape=h.delta_orbital_basis.shape, dtype=np.complex64)
    delta_init += (0.2 * rng.random(size=h.delta_orbital_basis.shape) - 1) + 1.0j * (
        0.2 * rng.random(size=h.delta_orbital_basis.shape) - 1
    )
    h.delta_orbital_basis = delta_init

    while True:
        new_gap = h.gap_equation(k=k_space_grid)
        if (np.abs(h.delta_orbital_basis - new_gap) < epsilon).all():
            h.delta_orbital_basis = new_gap
            return h
        mixing_greed = 0.2
        h.delta_orbital_basis = mixing_greed * new_gap + (1 - mixing_greed) * h.delta_orbital_basis
