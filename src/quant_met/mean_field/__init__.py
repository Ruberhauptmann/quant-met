# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""
Mean field treatment (:mod:`quant_met.mean_field`)
==================================================

Hamiltonians
------------

Base

.. autosummary::
   :toctree: generated/

    BaseHamiltonian

.. autosummary::
   :toctree: generated/

    GrapheneHamiltonian
    EGXHamiltonian


Functions
---------

.. autosummary::
   :toctree: generated/

   superfluid_weight
   quantum_metric
   free_energy
   free_energy_uniform_pairing
"""  # noqa: D205, D400

from .base_hamiltonian import BaseHamiltonian
from .eg_x import EGXHamiltonian
from .free_energy import (
    free_energy,
    free_energy_complex_gap,
    free_energy_real_gap,
    free_energy_uniform_pairing,
)
from .graphene import GrapheneHamiltonian
from .quantum_metric import quantum_metric, quantum_metric_bdg
from .self_consistency import self_consistency_loop
from .superfluid_weight import superfluid_weight

__all__ = [
    "superfluid_weight",
    "quantum_metric",
    "quantum_metric_bdg",
    "free_energy",
    "free_energy_complex_gap",
    "free_energy_real_gap",
    "free_energy_uniform_pairing",
    "self_consistency_loop",
    "BaseHamiltonian",
    "GrapheneHamiltonian",
    "EGXHamiltonian",
]
