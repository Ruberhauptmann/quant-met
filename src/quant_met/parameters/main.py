"""Pydantic models to hold parameters to run a simulation."""

from pydantic import BaseModel

from .control import Control


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
    k_points: KPoints
