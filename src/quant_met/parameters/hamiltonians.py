# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""
hamiltonians
============

Classes holding the configuration for the Hamiltonians.

.. autosummary::
   :toctree:
   :template: autosummary/pydantic.rst

    HamiltonianParameters
    DressedGrapheneParameters
    GrapheneParameters
    OneBandParameters
    TwoBandParameters
    ThreeBandParameters
"""  # noqa: D205, D400

from typing import Literal, TypeVar

import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo

GenericParameters = TypeVar("GenericParameters", bound="HamiltonianParameters")


def check_positive_values(value: float, info: ValidationInfo) -> float:
    """Check for positive values."""
    if value < 0:
        msg = f"{info.field_name} must be positive"
        raise ValueError(msg)
    return value


def validate_float(value: float, info: ValidationInfo) -> float:
    """Check for valid floats."""
    if np.isinf(value):
        msg = f"{info.field_name} must not be Infinity"
        raise ValueError(msg)
    if np.isnan(value):
        msg = f"{info.field_name} must not be NaN"
        raise ValueError(msg)
    return value


class HamiltonianParameters(BaseModel):
    """Base class for Hamiltonian parameters."""

    name: str
    """The name of the Hamiltonian model (e.g., "Graphene", "DressedGraphene")."""
    hubbard_int_orbital_basis: NDArray
    """A numpy array representing the Hubbard interactions in the orbital basis."""
    beta: float = np.inf
    """The inverse temperature; default is set to infinity."""
    q: NDArray[Shape["2"], float] | None = None
    """An optional numpy array representing the momentum of Cooper pairs."""


class DressedGrapheneParameters(HamiltonianParameters):
    """Parameters for the Dressed Graphene model.

    Attributes
    ----------
    hopping_gr : float
        Hopping parameter in the graphene layer.
    hopping_x : float
        Hopping parameter at the impurity site.
    hopping_x_gr_a : float
        Hybridization parameter.
    lattice_constant : float
        The lattice constant of the model.
    chemical_potential : float
        The chemical potential.
    hubbard_int_orbital_basis : npt.NDArray[np.float64]
        Hubbard interaction in the orbital basis.
    delta : npt.NDArray[np.complex64] | None
        Initial value for gaps in orbital space.
    """

    name: Literal["DressedGraphene"] = "DressedGraphene"
    hopping_gr: float = Field(..., description="Hopping in graphene")
    hopping_x: float = Field(..., description="Hopping in impurity")
    hopping_x_gr_a: float = Field(..., description="Hybridization")
    lattice_constant: float = Field(..., description="Lattice constant")
    chemical_potential: float = Field(..., description="Chemical potential")
    hubbard_int_orbital_basis: NDArray[Shape["3"], np.float64] = Field(
        ..., description="Hubbard interaction in orbital basis"
    )
    delta: NDArray[Shape["3"], np.complex64] | None = Field(
        default=None, description="Initial value for gaps in orbital space"
    )

    _check_positive_values = field_validator(
        "hopping_gr", "hopping_x", "hopping_x_gr_a", "lattice_constant"
    )(check_positive_values)

    _check_valid_floats = field_validator(
        "hopping_gr", "hopping_x", "hopping_x_gr_a", "lattice_constant", "chemical_potential"
    )(validate_float)


class GrapheneParameters(HamiltonianParameters):
    """Parameters for Graphene model."""

    name: Literal["Graphene"] = "Graphene"
    hopping: float
    lattice_constant: float
    chemical_potential: float
    hubbard_int_orbital_basis: NDArray[Shape["2"], np.float64] = Field(
        ..., description="Hubbard interaction in orbital basis"
    )
    delta: NDArray[Shape["2"], np.complex64] | None = None

    _check_positive_values = field_validator("hopping", "lattice_constant")(check_positive_values)

    _check_valid_floats = field_validator("hopping", "lattice_constant", "chemical_potential")(
        validate_float
    )


class OneBandParameters(HamiltonianParameters):
    """Parameters for Graphene model."""

    name: Literal["OneBand"] = "OneBand"
    hopping: float
    lattice_constant: float
    chemical_potential: float
    hubbard_int_orbital_basis: NDArray[Shape["1"], np.float64] = Field(
        ..., description="Hubbard interaction in orbital basis"
    )
    delta: NDArray[Shape["1"], np.complex64] | None = None

    _check_positive_values = field_validator("hopping", "lattice_constant")(check_positive_values)

    _check_valid_floats = field_validator("hopping", "lattice_constant", "chemical_potential")(
        validate_float
    )


class TwoBandParameters(HamiltonianParameters):
    """Parameters for Graphene model."""

    name: Literal["TwoBand"] = "TwoBand"
    hopping: float
    lattice_constant: float
    chemical_potential: float
    hubbard_int_orbital_basis: NDArray[Shape["2"], np.float64] = Field(
        ..., description="Hubbard interaction in orbital basis"
    )
    delta: NDArray[Shape["2"], np.complex64] | None = None

    _check_positive_values = field_validator("hopping", "lattice_constant")(check_positive_values)

    _check_valid_floats = field_validator("hopping", "lattice_constant", "chemical_potential")(
        validate_float
    )


class ThreeBandParameters(HamiltonianParameters):
    """Parameters for Graphene model."""

    name: Literal["ThreeBand"] = "ThreeBand"
    hopping: float
    lattice_constant: float
    chemical_potential: float
    hubbard_int_orbital_basis: NDArray[Shape["3"], np.float64] = Field(
        ..., description="Hubbard interaction in orbital basis"
    )
    delta: NDArray[Shape["3"], np.complex64] | None = None

    _check_positive_values = field_validator("hopping", "lattice_constant")(check_positive_values)

    _check_valid_floats = field_validator("hopping", "lattice_constant", "chemical_potential")(
        validate_float
    )
