"""
Bogoliubov-de Gennes (BdG)
==========================

.. autosummary::
    :toctree: generated/

    bdg_hamiltonian
    bdg_hamiltonian_derivative
    diagonalize_bdg
    gap_equation
"""  # noqa: D205, D400

from .bdg_hamiltonian import bdg_hamiltonian, bdg_hamiltonian_derivative, diagonalize_bdg
from .gap_equation import gap_equation

__all__ = ["bdg_hamiltonian", "gap_equation", "diagonalize_bdg", "bdg_hamiltonian_derivative"]
