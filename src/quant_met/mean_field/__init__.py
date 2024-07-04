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
from .free_energy import free_energy, free_energy_uniform_pairing
from .graphene import GrapheneHamiltonian
from .quantum_metric import quantum_metric
from .superfluid_weight import superfluid_weight

__all__ = [
    "superfluid_weight",
    "quantum_metric",
    "free_energy",
    "free_energy_uniform_pairing",
    "BaseHamiltonian",
    "GrapheneHamiltonian",
    "EGXHamiltonian",
]
