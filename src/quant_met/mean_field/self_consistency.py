# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT
"""Self-consistency loop."""

from .base_hamiltonian import BaseHamiltonian


def self_consistency_loop(h: BaseHamiltonian, epsilon: float) -> None:
    """Self-consistency loop.

    Parameters
    ----------
    h
    epsilon
    """
    del h
    del epsilon
