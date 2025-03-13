# SPDX-FileCopyrightText: 2025 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Utility functions used in DMFT."""

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
from mpi4py import MPI
from triqs.gf import Gf, MeshBrZone, MeshImFreq, MeshProduct, conjugate, dyson, inverse, iOmega_n


def _check_convergence(
    func: npt.NDArray[np.complex128], threshold: float = 1e-6, nsuccess: int = 1, nloop: int = 100
) -> tuple[float, bool]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    func = np.asarray(func)
    err = 1.0
    conv_bool = False
    outfile = "error.err"

    if globals().get("_whichiter") is None:
        global _whichiter
        global _gooditer
        global _oldfunc

        _whichiter = 0
        _gooditer = 0
        _oldfunc = np.zeros_like(func)

    green = "\033[92m"
    yellow = "\033[93m"
    red = "\033[91m"
    bold = "\033[1m"
    colorend = "\033[0m"

    # only the master does the calculation
    if rank == 0:
        errvec = np.real(np.sum(abs(func - _oldfunc), axis=-1) / np.sum(abs(func), axis=-1))
        # first iteration
        if _whichiter == 0:
            errvec = np.ones_like(errvec)
            # remove nan compoments, if some component is divided by zero
        if np.prod(np.shape(errvec)) > 1:
            errvec = errvec[~np.isnan(errvec)]
        errmax = np.max(errvec)
        errmin = np.min(errvec)
        err = np.average(errvec)
        _oldfunc = np.copy(func)
        if err < threshold:
            _gooditer += 1  # increase good iterations count
        else:
            _gooditer = 0  # reset good iterations count
        _whichiter += 1
        conv_bool = ((err < threshold) and (_gooditer > nsuccess) and (_whichiter < nloop)) or (
            _whichiter >= nloop
        )

        # write out
        with Path(outfile).open("a") as file:
            file.write(f"{_whichiter} {err:.6e}\n")
        if np.prod(np.shape(errvec)) > 1:
            with Path(outfile + ".max").open("a") as file:
                file.write(f"{_whichiter} {errmax:.6e}\n")
            with Path(outfile + ".min").open("a") as file:
                file.write(f"{_whichiter} {errmin:.6e}\n")
            with Path(outfile + ".distribution").open("a") as file:
                file.write(
                    f"{_whichiter}" + " ".join([f"{x:.6e}" for x in errvec.flatten()]) + "\n"
                )

        # print convergence message:
        if conv_bool:
            colorprefix = bold + green
        elif (err < threshold) and (_gooditer <= nsuccess):
            colorprefix = bold + yellow
        else:
            colorprefix = bold + red

        if _whichiter < nloop:
            if np.prod(np.shape(errvec)) > 1:
                print(colorprefix + "max error=" + colorend + f"{errmax:.6e}")
            print(
                colorprefix
                + "    " * (np.prod(np.shape(errvec)) > 1)
                + "error="
                + colorend
                + f"{err:.6e}"
            )
            if np.prod(np.shape(errvec)) > 1:
                print(colorprefix + "min error=" + colorend + f"{errmin:.6e}")
        else:
            if np.prod(np.shape(errvec)) > 1:
                print(colorprefix + "max error=" + colorend + f"{errmax:.6e}")
            print(
                colorprefix
                + "    " * (np.prod(np.shape(errvec)) > 1)
                + "error="
                + colorend
                + f"{err:.6e}"
            )
            if np.prod(np.shape(errvec)) > 1:
                print(colorprefix + "min error=" + colorend + f"{errmin:.6e}")
            print("Not converged after " + str(nloop) + " iterations.")
            with Path("ERROR.README").open("a") as file:
                file.write("Not converged after " + str(nloop) + " iterations.")
        print("\n")

    # pass to other cores:
    conv_bool = comm.bcast(conv_bool, root=0)
    err = comm.bcast(err, root=0)
    sys.stdout.flush()
    return err, conv_bool


def get_gloc(
    s: Gf,
    s_an: Gf,
    h0_nambu_k: Gf,
    xmu: npt.NDArray[np.complex128],
    broadening: float,
    kmesh: MeshBrZone,
) -> tuple[Gf, Gf]:
    """Compute local GF from bare lattice Hamiltonian and self-energy.

    Parameters
    ----------
    s
    s_an
    h0_nambu_k

    Returns
    -------
    tuple[Gf, Gf]

    """
    z = Gf(mesh=s.mesh, target_shape=h0_nambu_k.target_shape)
    n_orbitals = z.target_shape[0] // 2
    if isinstance(s.mesh, MeshImFreq):
        z[:n_orbitals, :n_orbitals] << iOmega_n + xmu - s
        z[:n_orbitals, n_orbitals:] << -s_an
        z[n_orbitals:, :n_orbitals] << -s_an
        z[n_orbitals:, n_orbitals:] << iOmega_n - xmu + conjugate(s)
    else:
        z[:n_orbitals, n_orbitals:] << -s_an
        z[n_orbitals:, :n_orbitals] << -s_an
        for w in z.mesh:
            z[w][:n_orbitals, :n_orbitals] = (w + 1j * broadening + xmu) * np.eye(n_orbitals) - s[w]
            z[w][n_orbitals:, n_orbitals:] = (w + 1j * broadening - xmu) * np.eye(
                n_orbitals
            ) + conjugate(s(-w))

    g_k = Gf(mesh=MeshProduct(kmesh, z.mesh), target_shape=h0_nambu_k.target_shape)
    for k in kmesh:
        g_k[k, :] << inverse(z - h0_nambu_k[k])

    g_loc_nambu = sum(g_k[k, :] for k in kmesh) / len(kmesh)

    g_loc = s.copy()
    g_loc_an = s_an.copy()
    g_loc[:] = g_loc_nambu[:n_orbitals, :n_orbitals]
    g_loc_an[:] = g_loc_nambu[:n_orbitals, n_orbitals:]
    return g_loc, g_loc_an


def _dmft_weiss_field(g_iw: Gf, g_an_iw: Gf, s_iw: Gf, s_an_iw: Gf) -> tuple[Gf, Gf]:
    """Compute Weiss field from local GF and self-energy.

    Parameters
    ----------
    g_iw
    g_an_iw
    s_iw
    s_an_iw

    Returns
    -------
    tuple[Gf, Gf]

    """
    n_orbitals = g_iw.target_shape[0]
    nambu_shape = (2 * n_orbitals, 2 * n_orbitals)
    g_nambu_iw = Gf(mesh=g_iw.mesh, target_shape=nambu_shape)
    s_nambu_iw = Gf(mesh=s_iw.mesh, target_shape=nambu_shape)

    g_nambu_iw[:n_orbitals, :n_orbitals] = g_iw
    g_nambu_iw[:n_orbitals, n_orbitals:] = g_an_iw
    g_nambu_iw[n_orbitals:, :n_orbitals] = g_an_iw
    g_nambu_iw[n_orbitals:, n_orbitals:] = -conjugate(g_iw)

    s_nambu_iw[:n_orbitals, :n_orbitals] = s_iw
    s_nambu_iw[:n_orbitals, n_orbitals:] = s_an_iw
    s_nambu_iw[n_orbitals:, :n_orbitals] = s_an_iw
    s_nambu_iw[n_orbitals:, n_orbitals:] = -conjugate(s_iw)

    g0_nambu_iw = dyson(G_iw=g_nambu_iw, Sigma_iw=s_nambu_iw)

    g0_iw = g_iw.copy()
    g0_an_iw = g_an_iw.copy()
    g0_iw[:] = g0_nambu_iw[:n_orbitals, :n_orbitals]
    g0_an_iw[:] = g0_nambu_iw[:n_orbitals, n_orbitals:]
    return g0_iw, g0_an_iw
