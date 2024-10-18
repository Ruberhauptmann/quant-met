# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""
Mean field treatment (:mod:`quant_met.mean_field`)
==================================================

Functions
---------

.. autosummary::
   :toctree: generated/

   superfluid_weight
   quantum_metric
"""  # noqa: D205, D400

from quant_met.mean_field import hamiltonians

from .quantum_metric import quantum_metric, quantum_metric_bdg
from .self_consistency import self_consistency_loop
from .superfluid_weight import superfluid_weight

__all__ = [
    "superfluid_weight",
    "quantum_metric",
    "quantum_metric_bdg",
    "self_consistency_loop",
    "hamiltonians",
]
