# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Pydantic models to hold parameters to run a simulation."""

from pydantic import BaseModel, Field

from .hamiltonians import DressedGrapheneParameters


class Control(BaseModel):
    """Control for the calculation."""

    calculation: str


class Parameters(BaseModel):
    """Class to hold the parameters for a calculation."""

    control: Control
    model: DressedGrapheneParameters = Field(..., discriminator="name")
