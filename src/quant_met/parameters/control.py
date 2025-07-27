"""Control parameters."""

import pathlib
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field

FloatList: TypeAlias = list[float]


class ControlBase(BaseModel):
    """Base class for control parameters."""

    prefix: str
    hamiltonian_file: pathlib.Path
    outdir: pathlib.Path
    conv_treshold: float
    hubbard_int_orbital_basis: FloatList = Field(..., min_length=1)
    max_iter: int = 1000


class SCF(ControlBase):
    """Parameters for the scf calculation."""

    calculation: Literal["scf"]
    beta: float
    calculate_additional: bool = False
    q: FloatList | None = Field(..., min_length=2, max_length=2)


class CritTemp(ControlBase):
    """Parameters for the critical temperature calculation."""

    calculation: Literal["crit-temp"]
    n_temp_points: int = 50
    q: FloatList | None = Field(..., min_length=2, max_length=2)


class QLoop(ControlBase):
    """Parameters for the q-loop calculation."""

    calculation: Literal["q-loop"]
    n_q_points: int = 50
    crit_temp: CritTemp | pathlib.Path


class QAnalysis(BaseModel):
    """Parameters for the q-analysis calculation."""

    calculation: Literal["q-analysis"]
    q_data: pathlib.Path
    hamiltonian_file: pathlib.Path
    prefix: str
    outdir: pathlib.Path


Control = Annotated[SCF | CritTemp | QLoop | QAnalysis, Field(discriminator="calculation")]
