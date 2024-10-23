.. SPDX-FileCopyrightText: 2024 Tjark Sievers
..
.. SPDX-License-Identifier: MIT

.. _changelog-0.0.8:

0.0.8 — 2024-10-23
------------------

Removed
^^^^^^^

- Functions to calculate free energy, as they are not needed anymore with the new self-consistency solver

Added
^^^^^

- Command-line-interface to run input files

- Finite momentum pairing into BdG Hamiltonian and self-consistency

- Finite momentum pairing into input file

- Function in Hamiltonian to calculate spectral gap from DOS

Changed
^^^^^^^

- Put Hamiltonians into subpackage under mean_field

Fixed
^^^^^

- Take lattice as argument in self-consistency, dont use Graphene lattice as default

.. _changelog-0.0.7:

0.0.7 — 2024-10-15
------------------

Added
^^^^^

- Function to calculate density of states from bands

Changed
^^^^^^^

- Multiply out phase factor of first entry in gap equation

Fixed
^^^^^

- Sum over bands for calculation of quantum metric in normal state as well

.. _changelog-0.0.6:

0.0.6 — 2024-10-07
------------------

Added
^^^^^

- Class bundling all aspects concerning lattice geometry

- Plotting methods for superfluid weight and quantum metric

- Proper self-consistent calculation of gap

- Implemented finite temperature into self-consistency calculation

- One band tight binding Hamiltonian

Changed
^^^^^^^

- Moved formatting of plots into a separate method

- Renamed variables in classes to be consistent and clearer

.. _changelog-0.0.5:

0.0.5 — 2024-08-27
------------------

Fixed
^^^^^

- Correct calculation of superfluid weight using the unitary matrix diagonalising the BdG Hamiltonian

.. _changelog-0.0.4:

0.0.4 — 2024-07-10
------------------

Added
^^^^^

- Implemented calculation of quantum metric for BdG states

Changed
^^^^^^^

- Hamiltonian methods now construct matrices in one turn from the whole k point list, this should significantly speed up calculations

.. _changelog-0.0.3:

0.0.3 — 2024-07-05
------------------

Added
^^^^^

- Add formula to calculate quantum metric

Changed
^^^^^^^

- Rename hamiltonians namespace to mean_field

- Implemented wrappers around the free energy calculation to calculate with a complex, real or uniform (in the orbitals) order parameter

- Calculate and return all components of the superfluid weight

.. _changelog-0.0.2:

0.0.2 — 2024-07-01
------------------

Added
^^^^^

- Can save and read results for a Hamiltonian, including parameters

- Calculation of superfluid weight

- Calculation of free energy at zero temperature

Changed
^^^^^^^

- Put units into plots

.. _changelog-0.0.1:

0.0.1 — 2024-05-31
------------------

Added
^^^^^

- Initial release with solid treatment of noninteracting models and gap equation ansatz
