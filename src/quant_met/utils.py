"""
Utility functions (:mod:`quant_met.utils`)
==========================================

.. currentmodule:: quant_met.utils

Functions
---------

.. autosummary::
   :toctree: generated/

    fermi_dirac
"""  # noqa: D205, D400

import numpy as np
import numpy.typing as npt
from numba import jit


@jit
def fermi_dirac(energy: npt.NDArray[np.floating], beta: float) -> npt.NDArray[np.floating]:
    """Fermi dirac distribution.

    Parameters
    ----------
    energy
    beta

    Returns
    -------
    fermi_dirac

    """
    return (
        np.where(energy < 0, 1.0, 0.0)
        if np.isinf(beta)
        else np.asarray(1 / (1 + np.exp(beta * energy)))
    )
