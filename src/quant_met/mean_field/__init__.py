# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""
Mean field
==========

Submodules
----------

.. autosummary::
    :toctree: generated/

    hamiltonians


Functions
---------

.. autosummary::
   :toctree: generated/

   superfluid_weight
   quantum_metric
   self_consistency_loop
   search_crit_temp
"""  # noqa: D205, D400

from quant_met.mean_field import hamiltonians

from .quantum_metric import quantum_metric
from .search_crit_temp import search_crit_temp
from .self_consistency import self_consistency_loop
from .superfluid_weight import superfluid_weight

__all__ = [
    "superfluid_weight",
    "quantum_metric",
    "self_consistency_loop",
    "search_crit_temp",
    "hamiltonians",
]
