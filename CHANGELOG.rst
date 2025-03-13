.. SPDX-FileCopyrightText: 2024 Tjark Sievers
..
.. SPDX-License-Identifier: MIT

.. _changelog-0.0.24:

0.0.24 — 2025-03-13
-------------------

Changed
^^^^^^^

- Add orbital positions into the Fourier transform

- Implemented a more efficient version of the gap equation using direkt matrix multiplication

Fixed
^^^^^

- Fix orbital positions in TBLattice for decorated Graphene

- Fix definition of TBLattice for dressed Graphene

.. _changelog-0.0.23:

0.0.23 — 2025-03-10
-------------------

Fixed
^^^^^

- Fix wrong chemical potential for Hubbard interaction in DMFT

.. _changelog-0.0.22:

0.0.22 — 2025-03-07
-------------------

Fixed
^^^^^

- Fix one wrong conjugate in calculation of SF weight

.. _changelog-0.0.21:

0.0.21 — 2025-03-07
-------------------

Changed
^^^^^^^

- Moved calculation of quantum metric into the Hamiltonian


.. _changelog-0.0.20:

0.0.20 — 2025-03-07
-------------------

Added
^^^^^

- First implementation of EDIpack library for DMFT

Fixed
^^^^^

- Multiplied the current in y direction in mean field instead of summing it up

.. _changelog-0.0.19:

0.0.19 — 2025-02-28
-------------------

Fixed
^^^^^

- Routine for finding T_C now work even for small absolute values of the gaps

.. _changelog-0.0.18:

0.0.18 — 2025-01-10
-------------------

Added
^^^^^

- Control parameter to calculate SC current, superfluid weight and free energy in scf calculation

.. _changelog-0.0.17:

0.0.17 — 2025-01-10
-------------------

Added
^^^^^

- Method to calculate free energy for a given hamiltonian

- Method to calculate current density for a given Hamiltonian

Changed
^^^^^^^

- Speedup of calculation of superfluid weight via caching of intermediate values

Fixed
^^^^^

- Sign in calculation of superfluid weight

.. _changelog-0.0.16:

0.0.16 — 2024-11-22
-------------------

Fixed
^^^^^

- Fit for critical temperatures

.. _changelog-0.0.15:

0.0.15 — 2024-11-21
-------------------

Fixed
^^^^^

- Fixed some cases of the crit_temp routine going into the wrong direction

.. _changelog-0.0.14:

0.0.14 — 2024-11-20
-------------------

Changed
^^^^^^^

- crit-temp routine now saves a sample Hamiltonian besides the critical temperatures

Fixed
^^^^^

- Search for T_C bounds, so that it does not loop anymore in certain cases

.. _changelog-0.0.13:

0.0.13 — 2024-11-19
-------------------

Added
^^^^^

- Routine to search for transition temperature

.. _changelog-0.0.12:

0.0.12 — 2024-11-09
-------------------

Changed
^^^^^^^

- Use numpy allclose function in covergence criterium

.. _changelog-0.0.11:

0.0.11 — 2024-11-08
-------------------

Changed
^^^^^^^

- Option to set maximum number of iterations in self-consistency loop

- Convergence criterium changed to be relative, i.e. the change in gap components is divided by the old gap components and then compared to the epsilon

.. _changelog-0.0.10:

0.0.10 — 2024-11-06
-------------------

Added
^^^^^

- Gap equation at zero temperature

- Proper logging and debug mode

Fixed
^^^^^

- Typing in Hamiltonian classes, so the from_file method returns the corresponding subclass

.. _changelog-0.0.9:

0.0.9 — 2024-10-28
------------------

Changed
^^^^^^^

- Save all simulation parameters into the output file

- Restructured mean_field Hamiltonian classes, so more functionality is concentrated in the base class

Fixed
^^^^^

- Fixed mistake in gap equation: had the algebra wrong, leading to the self-consistency not converging correctly

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
