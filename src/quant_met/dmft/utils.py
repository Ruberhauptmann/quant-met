# SPDX-FileCopyrightText: 2025 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Utility functions used in DMFT."""

import numpy as np
import numpy.typing as npt
from triqs.gf import Gf, MeshBrZone, MeshImFreq, MeshProduct, conjugate, dyson, inverse, iOmega_n


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
