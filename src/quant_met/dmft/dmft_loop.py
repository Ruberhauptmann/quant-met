# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Functions to run self-consistent calculation for the order parameter."""

import logging
from itertools import product

import numpy as np
import numpy.typing as npt
from edipack2triqs.fit import BathFittingParams
from edipack2triqs.solver import EDIpackSolver
from triqs.gf import BlockGf, Gf, MeshBrZone
from triqs.lattice.tight_binding import TBLattice
from triqs.operators import c, c_dag, dagger, n

from quant_met.mean_field.hamiltonians import BaseHamiltonian
from quant_met.parameters import GenericParameters

from .utils import _check_convergence, _dmft_weiss_field, get_gloc

logger = logging.getLogger(__name__)


def dmft_loop(
    tbl: TBLattice,
    h: BaseHamiltonian[GenericParameters],
    h0_nambu_k: Gf,
    n_bath: float,
    n_iw: int,
    broadening: float,
    n_w: int,
    w_mixing: float,
    n_success: int,
    xmu: npt.NDArray[np.float64],
    kmesh: MeshBrZone,
    epsilon: float,
    max_iter: int,
) -> EDIpackSolver:
    """DMFT loop.

    Parameters
    ----------
    tbl
    h
    h0_nambu_k
    n_bath
    n_iw
    broadening
    n_w
    w_mixing
    n_success
    xmu
    kmesh
    epsilon
    max_iter

    Returns
    -------
    EDIpackSolver

    """
    energy_window = (-2.0 * h.hopping_gr, 2.0 * h.hopping_gr)

    spins = ("up", "dn")
    orbs = range(tbl.n_orbitals)

    # Fundamental sets for impurity degrees of freedom
    fops_imp_up = [("up", o) for o in orbs]
    fops_imp_dn = [("dn", o) for o in orbs]

    # Fundamental sets for bath degrees of freedom
    fops_bath_up = [("B_up", i) for i in range(tbl.n_orbitals * n_bath)]
    fops_bath_dn = [("B_dn", i) for i in range(tbl.n_orbitals * n_bath)]

    # Non-interacting part of the impurity Hamiltonian
    h_loc = -xmu * np.eye(tbl.n_orbitals)
    hamiltonian = sum(
        h_loc[o1, o2] * c_dag(spin, o1) * c(spin, o2) for spin, o1, o2 in product(spins, orbs, orbs)
    )

    ust = 0
    jh = 0
    jx = 0
    jp = 0

    # Interaction part
    hamiltonian += h.hubbard_int_orbital_basis[0] * sum(n("up", o) * n("dn", o) for o in orbs)
    hamiltonian += ust * sum(
        int(o1 != o2) * n("up", o1) * n("dn", o2) for o1, o2 in product(orbs, orbs)
    )
    hamiltonian += (ust - jh) * sum(
        int(o1 < o2) * n(s, o1) * n(s, o2) for s, o1, o2 in product(spins, orbs, orbs)
    )
    hamiltonian -= jx * sum(
        int(o1 != o2) * c_dag("up", o1) * c("dn", o1) * c_dag("dn", o2) * c("up", o2)
        for o1, o2 in product(orbs, orbs)
    )
    hamiltonian += jp * sum(
        int(o1 != o2) * c_dag("up", o1) * c_dag("dn", o1) * c("dn", o2) * c("up", o2)
        for o1, o2 in product(orbs, orbs)
    )

    # Matrix dimensions of eps and V: 3 orbitals x 2 bath states
    eps = np.array([[-1.0, -0.5, 0.5, 1.0] for _ in range(tbl.n_orbitals)])
    v = 0.5 * np.ones((tbl.n_orbitals, n_bath))
    d = -0.2 * np.eye(tbl.n_orbitals * n_bath)

    # Bath
    hamiltonian += sum(
        eps[o, nu] * c_dag("B_" + s, o * n_bath + nu) * c("B_" + s, o * n_bath + nu)
        for s, o, nu in product(spins, orbs, range(n_bath))
    )

    hamiltonian += sum(
        v[o, nu]
        * (c_dag(s, o) * c("B_" + s, o * n_bath + nu) + c_dag("B_" + s, o * n_bath + nu) * c(s, o))
        for s, o, nu in product(spins, orbs, range(n_bath))
    )

    # Anomalous bath
    hamiltonian += sum(
        d[o, q] * (c("B_up", o) * c("B_dn", q)) + dagger(d[o, q] * (c("B_up", o) * c("B_dn", q)))
        for o, q in product(range(tbl.n_orbitals * n_bath), range(tbl.n_orbitals * n_bath))
    )

    # Create solver object
    fit_params = BathFittingParams(method="minimize", grad="numeric")
    solver = EDIpackSolver(
        hamiltonian,
        fops_imp_up,
        fops_imp_dn,
        fops_bath_up,
        fops_bath_dn,
        lanc_dim_threshold=1024,
        verbose=1,
        bath_fitting_params=fit_params,
    )

    for iloop in range(max_iter):
        print(f"\nLoop {iloop + 1} of {max_iter}")

        # Solve the effective impurity problem
        solver.solve(
            beta=h.beta,
            n_iw=n_iw,
            energy_window=energy_window,
            n_w=n_w,
            broadening=broadening,
        )

        # Normal and anomalous components of computed self-energy
        s_iw = solver.Sigma_iw["up"]
        s_an_iw = solver.Sigma_an_iw["up_dn"]

        # Compute local Green's function
        g_iw, g_an_iw = get_gloc(s_iw, s_an_iw, h0_nambu_k, xmu, broadening, kmesh)
        # Compute Weiss field
        g0_iw, g0_an_iw = _dmft_weiss_field(g_iw, g_an_iw, s_iw, s_an_iw)

        # Bath fitting and mixing
        g0_iw_full = BlockGf(name_list=spins, block_list=[g0_iw, g0_iw])
        g0_an_iw_full = BlockGf(name_list=["up_dn"], block_list=[g0_an_iw])

        bath_new = solver.chi2_fit_bath(g0_iw_full, g0_an_iw_full)[0]
        solver.bath = w_mixing * bath_new + (1 - w_mixing) * solver.bath

        # Check convergence of the Weiss field
        g0 = np.asarray([g0_iw.data, g0_an_iw.data])
        # Check convergence of the Weiss field
        g0 = np.asarray([g0_iw.data, g0_an_iw.data])
        err, converged = _check_convergence(g0, epsilon, n_success, max_iter)

        if converged:
            break

    return solver
