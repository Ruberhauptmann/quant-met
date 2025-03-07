# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Pydantic models to hold parameters to run a simulation."""

import pathlib

from pydantic import BaseModel, Field

from .hamiltonians import (
    DressedGrapheneParameters,
    GrapheneParameters,
    OneBandParameters,
    ThreeBandParameters,
    TwoBandParameters,
)


class Control(BaseModel):
    """Control for the calculation.

    Attributes
    ----------
    calculation : str
        Specifies the type of calculation to be performed.
    prefix : str
        A string used as a prefix for naming output files generated by the simulation.
    outdir : :class:`pathlib.Path`
        A path indicating the output directory where results will be saved.
    conv_treshold : float
        A float value representing the convergence threshold.
        The calculation will stop when changes in the results drop below this threshold.
    """

    calculation: str
    prefix: str
    outdir: pathlib.Path
    conv_treshold: float
    max_iter: int = 1000
    n_temp_points: int = 50
    calculate_additional: bool = False

    n_spin: int = 1
    n_success: int = 1
    wmixing: float = 0.5
    n_bath: int = 2
    n_iw: int = 1024
    n_w: int = 4000
    broadening: float = 0.005


class KPoints(BaseModel):
    """Control for k points.

    Attributes
    ----------
    nk1 : int
        The number of k-points in the first dimension of the k-space grid.
    nk2 : int
        The number of k-points in the second dimension of the k-space grid.
    """

    nk1: int
    nk2: int


class Parameters(BaseModel):
    """Class to hold the parameters for a calculation.

    Attributes
    ----------
    control : Control
        An instance of the `Control` class containing settings for the calculation.
    model :
        An instance of one of the Hamiltonian parameter classes, holding the specific parameters
        of the selected Hamiltonian model.
    k_points : KPoints
        An instance of the `KPoints` class that specifies the number of k-points for the simulation.
    """

    control: Control
    model: (
        DressedGrapheneParameters
        | GrapheneParameters
        | OneBandParameters
        | TwoBandParameters
        | ThreeBandParameters
    ) = Field(..., discriminator="name")
    k_points: KPoints
