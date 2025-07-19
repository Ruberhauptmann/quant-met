import pathlib
from typing import Annotated, Literal

from pydantic import BaseModel, Field, conlist

HubbardInt = Annotated[
    conlist(float, min_length=1), Field(..., description="Hubbard interaction in orbital basis")
]
QVector = Annotated[conlist(float, min_length=2, max_length=2), Field(..., description="q vector")]


class ControlBase(BaseModel):
    """Base class for control parameters."""

    prefix: str
    hamiltonian_file: pathlib.Path
    outdir: pathlib.Path
    conv_treshold: float
    hubbard_int_orbital_basis: HubbardInt
    max_iter: int = 1000


class SCF(ControlBase):
    """Parameters for the scf calculation."""

    calculation: Literal["scf"]
    beta: float
    calculate_additional: bool = False
    q: QVector | None = None


class CritTemp(ControlBase):
    """Parameters for the critical temperature calculation."""

    calculation: Literal["crit-temp"]
    n_temp_points: int = 50
    q: QVector | None = None


class QLoop(ControlBase):
    """Parameters for the q-loop calculation."""

    calculation: Literal["q-loop"]
    n_q_points: int = 50
    crit_temp: CritTemp | pathlib.Path


Control = Annotated[SCF | CritTemp | QLoop, Field(discriminator="calculation")]
