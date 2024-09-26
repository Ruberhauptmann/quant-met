# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Lattice geometry for Square Lattice."""

import numpy as np


class SquareLattice:
    """Lattice geometry for Square Lattice."""

    lattice_constant = 1
    bz_corners = (
        np.pi
        / lattice_constant
        * np.array([np.array([1, 1]), np.array([-1, 1]), np.array([1, -1]), np.array([-1, -1])])
    )

    Gamma = np.array([0, 0])
    M = np.pi / lattice_constant * np.array([1, 1])
    X = np.pi / lattice_constant * np.array([1, 0])

    high_symmetry_points = ((Gamma, r"\Gamma"), (M, "M"))
