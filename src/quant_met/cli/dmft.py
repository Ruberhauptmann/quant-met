# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Functions to run self-consistent calculation for the order parameter."""

import logging
from pathlib import Path

from h5 import HDFArchive
from mpi4py import MPI
from triqs.gf import Gf

from quant_met.cli._utils import _hamiltonian_factory, _tbl_factory
from quant_met.dmft.dmft_loop import dmft_loop
from quant_met.dmft.utils import get_gloc
from quant_met.parameters import Parameters

logger = logging.getLogger(__name__)


def dmft_scf(parameters: Parameters) -> None:
    """Self-consistent calculation for the order parameter.

    Parameters
    ----------
    parameters: Parameters
        An instance of Parameters containing control settings, the model,
        and k-point specifications for the self-consistency calculation.
    """
    result_path = Path(parameters.control.outdir)
    result_path.mkdir(exist_ok=True, parents=True)

    h = _hamiltonian_factory(parameters=parameters.model, classname=parameters.model.name)
    tbl = _tbl_factory(h=h)

    kmesh = tbl.get_kmesh(n_k=(parameters.k_points.nk1, parameters.k_points.nk2, 1))

    enk = tbl.fourier(kmesh)
    n_orbitals = tbl.n_orbitals
    nambu_shape = (2 * n_orbitals, 2 * n_orbitals)
    h0_nambu_k = Gf(mesh=kmesh, target_shape=nambu_shape)
    for k in kmesh:
        h0_nambu_k[k][:n_orbitals, :n_orbitals] = enk(k)
        h0_nambu_k[k][n_orbitals:, n_orbitals:] = -enk(-k)

    ust = 0
    jh = 0
    xmu = (
        h.hubbard_int_orbital_basis[0] / 2
        + (tbl.n_orbitals - 1) * ust / 2
        + (tbl.n_orbitals - 1) * (ust - jh) / 2
    )

    solver = dmft_loop(
        tbl=tbl,
        h=h,
        h0_nambu_k=h0_nambu_k,
        n_bath=parameters.control.n_bath,
        n_iw=parameters.control.n_iw,
        broadening=parameters.control.broadening,
        n_w=parameters.control.n_w,
        w_mixing=parameters.control.wmixing,
        n_success=parameters.control.n_success,
        xmu=xmu,
        kmesh=kmesh,
        epsilon=parameters.control.conv_treshold,
        max_iter=parameters.control.max_iter,
    )

    # Calculate local Green's function on the real axis
    s_w = solver.Sigma_w["up"]
    s_an_w = solver.Sigma_an_w["up_dn"]
    s_iw = solver.Sigma_iw["up"]
    s_an_iw = solver.Sigma_an_iw["up_dn"]
    g_iw, g_an_iw = get_gloc(s_iw, s_an_iw, h0_nambu_k, xmu, parameters.control.broadening, kmesh)
    g_w, g_an_w = get_gloc(s_w, s_an_w, h0_nambu_k, xmu, parameters.control.broadening, kmesh)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        data_dir = Path("data/DressedGraphene/dmft/sweep_V/")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save calculation results
        result_file = result_path / f"{parameters.control.prefix}.hdf5"
        with HDFArchive(f"{result_file}", "w") as ar:
            ar["s_iw"] = s_iw
            ar["s_an_iw"] = s_an_iw
            ar["g_iw"] = g_iw
            ar["g_an_iw"] = g_an_iw
            ar["g_w"] = g_w
            ar["g_an_w"] = g_an_w

        logger.info("Results saved to %s", result_file)
