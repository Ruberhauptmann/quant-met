.. SPDX-FileCopyrightText: 2024 Tjark Sievers
..
.. SPDX-License-Identifier: MIT

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
