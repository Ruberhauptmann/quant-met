# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Pydantic models to hold parameters for Hamiltonians."""

from typing import Literal

import numpy as np
from pydantic import BaseModel
from pydantic_numpy import np_array_pydantic_annotated_typing


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
    delta: np_array_pydantic_annotated_typing(data_type=np.complex64, dimensions=1) | None = None
