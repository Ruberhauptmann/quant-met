# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Pydantic models to hold parameters for Hamiltonians."""

from typing import Literal

import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel


class DressedGrapheneParameters(BaseModel):
    """Parameters for the dressed Graphene model."""

    name: Literal["DressedGraphene"] = "DressedGraphene"
    hopping_gr: float
    hopping_x: float
    hopping_x_gr_a: float
    lattice_constant: float
    chemical_potential: float
    hubbard_int_gr: float
    hubbard_int_x: float
    delta: NDArray[Shape["3"], np.complex64] | None = None


class GrapheneParameters(BaseModel):
    """Parameters for Graphene model."""

    name: Literal["Graphene"] = "Graphene"
    hopping: float
    lattice_constant: float
    chemical_potential: float
    hubbard_int: float
    delta: NDArray[Shape["2"], np.complex64] | None = None


class OneBandParameters(BaseModel):
    """Parameters for Graphene model."""

    name: Literal["OneBand"] = "OneBand"
    hopping: float
    lattice_constant: float
    chemical_potential: float
    hubbard_int: float
    delta: NDArray[Shape["1"], np.complex64] | None = None
