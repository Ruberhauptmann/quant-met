# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Pydantic models to hold parameters to run a simulation."""

import pathlib

from numpydantic import NDArray, Shape
from pydantic import BaseModel, Field

from .hamiltonians import DressedGrapheneParameters, GrapheneParameters, OneBandParameters


class Control(BaseModel):
    """Control for the calculation."""

    calculation: str
    prefix: str
    outdir: pathlib.Path
    conv_treshold: float
    beta: float
    q: NDArray[Shape["2"], int | float] | None = None


class KPoints(BaseModel):
    """Control for k points."""

    nk1: int
    nk2: int


class Parameters(BaseModel):
    """Class to hold the parameters for a calculation."""

    control: Control
    model: DressedGrapheneParameters | GrapheneParameters | OneBandParameters = Field(
        ..., discriminator="name"
    )
    k_points: KPoints
