# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Functions to run self-consistent calculation for the order parameter."""

import logging
from itertools import product

import numpy as np
import numpy.typing as npt
from triqs.gf import BlockGf, Gf, MeshBrZone
from triqs.lattice.tight_binding import TBLattice
from triqs.operators import c, c_dag, dagger, n

from edipack2triqs.fit import BathFittingParams
from edipack2triqs.solver import EDIpackSolver
from quant_met.mean_field.hamiltonians import BaseHamiltonian
from quant_met.parameters import GenericParameters

from .utils import _dmft_weiss_field, get_gloc

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

    # Interaction part
    hamiltonian += -h.hubbard_int_orbital_basis[0] * sum(n("up", o) * n("dn", o) for o in orbs)

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

    gooditer = 0
    g0_prev = np.zeros((2, 2 * n_iw, tbl.n_orbitals, tbl.n_orbitals), dtype=complex)
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
        errvec = np.real(np.sum(abs(g0 - g0_prev), axis=1) / np.sum(abs(g0), axis=1))
        # First iteration
        if iloop == 0:
            errvec = np.ones_like(errvec)
        errmin, err, errmax = np.min(errvec), np.average(errvec), np.max(errvec)

        g0_prev = np.copy(g0)

        if err < epsilon:
            gooditer += 1  # Increase good iterations count
        else:
            gooditer = 0  # Reset good iterations count

        conv_bool = ((err < epsilon) and (gooditer > n_success) and (iloop < max_iter)) or (
            iloop >= max_iter
        )

        # Print convergence message
        if iloop < max_iter:
            if errvec.size > 1:
                print(f"max error={errmax:.6e}")
            print("    " * (errvec.size > 1) + f"error={err:.6e}")
            if errvec.size > 1:
                print(f"min error={errmin:.6e}")
        else:
            if errvec.size > 1:
                print(f"max error={errmax:.6e}")
            print("    " * (errvec.size > 1) + f"error={err:.6e}")
            if errvec.size > 1:
                print(f"min error={errmin:.6e}")
            print(f"Not converged after {max_iter} iterations.")

        if conv_bool:
            break

    return solver
