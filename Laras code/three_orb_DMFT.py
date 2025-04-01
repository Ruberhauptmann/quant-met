#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Tjark Sievers
#
# SPDX-License-Identifier: MIT

import os
from itertools import product

import numpy as np
from aux_funx import *
from mpi4py import MPI
from triqs.plot.mpl_interface import oplot, plt

result_dir = sys.argv[1]
os.chdir(result_dir)

# TRIQS modules
from edipack2triqs.fit import BathFittingParams

# edipack2triqs modules
from edipack2triqs.solver import EDIpackSolver
from h5 import HDFArchive
from triqs.operators import c, c_dag, n

# INIT MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == comm.Get_size:
    print("I am process", rank, "of", comm.Get_size())
master = rank == 0

# Parameters
Nspin = 1
Nloop = 50
Nsuccess = 1
threshold = 1e-5
wmixing = 0.8
Norb = 3
Nbath = 2
U = float(sys.argv[4])  # Local intra-orbital interactions U
Jh = float(sys.argv[5])  # Hund's coupling
Ust = U - 2 * Jh  # Local inter-orbital interaction U'
Jx = 0.0  # Spin-exchange couplsng constant
Jp = 0.0  # Pair-hopping coupling constant
xmu = U / 2 + (Norb - 1) * Ust / 2 + (Norb - 1) * (Ust - Jh) / 2  # Chemical potential
beta = float(sys.argv[3])  # Inverse temperature
n_iw = 1024  # Number of Matsubara frequencies for impurity GF calculations
maxbath = 2.0  # maximum bath energy
t = 0.5
bath_reload = sys.argv[2]
B_site = False
lanc_nstates_total = 120
lanc_nstates_sector = 10
Symmetrize_bath = True

offset = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -0.0]])

# Energy window for real-frequency GF calculations
energy_window = (-5 * t, 5 * t)
n_w = 4000  # Number of real-frequency points for impurity GF calculations
broadening = 0.05  # Broadening on the real axis for impurity GF calculations

if master:
    ar = HDFArchive("results.h5", "w")
    params = {
        k: v
        for k, v in globals().items()
        if not k.startswith("__") and isinstance(v, (int, float, bool, str))
    }
    for key, value in params.items():
        ar[key] = value

    ar.create_group("observables")
    ar.create_group("convergence")
    ar.create_group("bath")
    ar.create_group("GFs")
    del ar


spins = ("up", "dn")
orbs = range(Norb)

# Fundamental sets for impurity degrees of freedom
fops_imp_up = [("up", o) for o in orbs]
fops_imp_dn = [("dn", o) for o in orbs]

# Fundamental sets for bath degrees of freedom
fops_bath_up = [("B_up", i) for i in range(Norb * Nbath)]
fops_bath_dn = [("B_dn", i) for i in range(Norb * Nbath)]

# Non-interacting part of the impurity Hamiltonrelian
h_loc = -xmu * np.eye(Norb) + offset
H = sum(
    h_loc[o1, o2] * c_dag(spin, o1) * c(spin, o2) for spin, o1, o2 in product(spins, orbs, orbs)
)


# Interaction part
H += U * sum(n("up", o) * n("dn", o) for o in orbs)
H += Ust * sum(int(o1 != o2) * n("up", o1) * n("dn", o2) for o1, o2 in product(orbs, orbs))
H += (Ust - Jh) * sum(
    int(o1 < o2) * n(s, o1) * n(s, o2) for s, o1, o2 in product(spins, orbs, orbs)
)
H -= Jx * sum(
    int(o1 != o2) * c_dag("up", o1) * c("dn", o1) * c_dag("dn", o2) * c("up", o2)
    for o1, o2 in product(orbs, orbs)
)
H += Jp * sum(
    int(o1 != o2) * c_dag("up", o1) * c_dag("dn", o1) * c("dn", o2) * c("up", o2)
    for o1, o2 in product(orbs, orbs)
)

# Rekoad bath if path is given, otherwise set evenly distributed bath to start
if bath_reload != "False":
    ar = HDFArchive(bath_reload, "r")
    eps = ar["bath"]["A_bath_eps"][-1, :, :]
    V = ar["bath"]["A_bath_V"][-1, :, :]
    if master:
        print("Loaded bath")
    del ar
else:
    eps = np.array(
        [
            [i + 0.001 * (abs(i) < 0.001) for i in np.linspace(-maxbath, maxbath, Nbath)]
            for j in range(Norb)
        ]
    )
    V = 0.71067811865 * np.ones((Norb, Nbath))


# Bath Hamiltonian
H += sum(
    eps[o, nu] * c_dag("B_" + s, o * Nbath + nu) * c("B_" + s, o * Nbath + nu)
    for s, o, nu in product(spins, orbs, range(Nbath))
)

H += sum(
    V[o, nu]
    * (c_dag(s, o) * c("B_" + s, o * Nbath + nu) + c_dag("B_" + s, o * Nbath + nu) * c(s, o))
    for s, o, nu in product(spins, orbs, range(Nbath))
)


# Parameters for fitting
fit_params = BathFittingParams(
    method="minimize", grad="numeric", scheme="delta", weight="1", n_iw=100
)
# Create solver object
solver = EDIpackSolver(
    H,
    fops_imp_up,
    fops_imp_dn,
    fops_bath_up,
    fops_bath_dn,
    lanc_dim_threshold=1024,
    verbose=2,
    lanc_nstates_total=lanc_nstates_total,
    lanc_nstates_sector=lanc_nstates_sector,
    lanc_nstates_step=10,
    ed_total_ud=False,
    lanc_niter=30000,
    lanc_ngfiter=300,
    bath_fitting_params=fit_params,
)


#############
# DMFT loop #
#############

converged = False
iloop = 1

while not converged and iloop < Nloop:
    if master:
        print(f"\nLoop {iloop + 1} of {Nloop}")

    # Solve the effective impurity problem
    solver.solve(beta=beta, n_iw=n_iw, energy_window=energy_window, n_w=n_w, broadening=broadening)

    # Normal components of computed g and self-energy
    s_iw = solver.Sigma_iw
    g_iw = solver.g_iw
    # Compute Weiss field
    delta_iw = g_iw.copy()

    if B_site is True:
        g_iw_B = g_iw.copy()

        for s in ("up", "dn"):

            def index_swap(i):
                assert i == 0 or i == 1 or i == 2
                if i == 0:
                    return 1
                if i == 1:
                    return 0
                if i == 2:
                    return 2

            for i, j in product(range(Norb), repeat=2):
                g_iw_B[s][i, j] = g_iw[s][index_swap(i), index_swap(j)]

            delta_iw << t**2 * g_iw_B
            if master:
                print("Self-consitency with virtual site B")
    if B_site is False:
        delta_iw << t**2 * g_iw
        if master:
            print("Self-consitency without virtual site B")

    if master:
        save_observables(site="A", result_dir=result_dir, Norb=Norb)
        # Save bath in hdf5 file
        ar = HDFArchive("results.h5", "a")
        if iloop == 1:
            ar["bath"]["A_bath_eps"] = solver.bath.eps
            ar["bath"]["A_bath_V"] = solver.bath.V
        else:
            ar["bath"]["A_bath_eps"] = np.concatenate(
                [ar["bath"]["A_bath_eps"], solver.bath.eps], axis=0
            )
            ar["bath"]["A_bath_V"] = np.concatenate([ar["bath"]["A_bath_V"], solver.bath.V], axis=0)
        ar["GFs"]["A_s_iw"] = s_iw
        ar["GFs"]["A_g_iw"] = g_iw
        ar["GFs"]["A_delta_iw"] = delta_iw
        del ar
    comm.Barrier()

    # Check convergence of the Weiss field
    tocheck = np.asarray([delta_iw["up"].data])
    err, converged = check_convergence(tocheck, threshold=threshold, Nsuccess=Nsuccess, Nloop=Nloop)

    if master:
        ar = HDFArchive("results.h5", "a")
        if iloop == 1:
            convergence_group = ar["convergence"]
            convergence_group["err"] = np.array([err])
        else:
            ar["convergence"]["err"] = np.append(ar["convergence"]["err"], err)
        ar["convergence"]["converged"] = converged
        del ar

    # Fit new bath
    bath_new, fitted_delta_normal = solver.chi2_fit_bath(delta_iw)

    """
    if Symmetrize_bath is True:
        for orb in range(Norb):
            for b1, b2 in zip(range(Nbath // 2), reversed(range(Nbath - Nbath // 2, Nbath))):
                eps_val1 = abs(bath_new.eps[0, orb, b1])
                eps_val2 = abs(bath_new.eps[0, orb, b2])
                avg_eps = 0.5 * (eps_val1 + eps_val2)
                bath_new.eps[0, orb, b1] = -avg_eps
                bath_new.eps[0, orb, b2] = avg_eps
                V_val1 = abs(bath_new.V[0, orb, b1])
                V_val2 = abs(bath_new.V[0, orb, b2])
                avg_V = 0.5 * (V_val1 + V_val2)
                bath_new.V[0, orb, b1] = avg_V
                bath_new.V[0, orb, b2] = avg_V

        # Fif n_bath uneven: set close to zero
        if Nbath % 2 == 1:
            middle = Nbath // 2
            eps[0, orb, middle] = 1e-5

        """

    # Check the fit
    for i in range(3):
        fit_directory = f"{result_dir}/check_fit"
        os.makedirs(fit_directory, exist_ok=True)
        oplot(delta_iw["up"][i, i].imag, "b.--", name=f"Im(delta_{i}{i})")
        oplot(delta_iw["up"][i, i].real, "g.--", name=f"Re(delta_{i}{i})")
        oplot(fitted_delta_normal["up"][i, i].imag, "r.--", name=f"Im(delta_{i}{i}_fit)")
        oplot(fitted_delta_normal["up"][i, i].real, "m.--", name=f"Re(delta_{i}{i}_fit)")
        plt.xlim(0.0, 4)
        plt.ylabel("Hybridization")

        # Save the plot
        output_file = f"check_fit/loop={iloop}_fit_orb={i}.pdf"
        plt.savefig(output_file)
        plt.close()

    # Mixing the bath
    solver.bath = (1 - wmixing) * bath_new + (wmixing) * solver.bath

    # Copy temporary folders
    comm.Barrier()
    if master:
        copy_temp_folder(result_dir=result_dir)
    comm.Barrier()

    # Overwiev plot
    if master:
        plot_run_overview_A(result_dir=result_dir, Nbath=Nbath, Norb=Norb)
    comm.Barrier()

    iloop += 1

if master:
    print("Done...")
