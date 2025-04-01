# SPDX-FileCopyrightText: 2025 Tjark Sievers
#
# SPDX-License-Identifier: MIT

import os
import sys
from io import StringIO

import numpy as np
import pandas as pd
from h5 import *
from triqs.plot.mpl_interface import plt


def check_convergence(func, threshold=1e-6, Nsuccess=1, Nloop=100):
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except:
        rank = 0

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

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    COLOREND = "\033[0m"

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
        conv_bool = ((err < threshold) and (_gooditer > Nsuccess) and (_whichiter < Nloop)) or (
            _whichiter >= Nloop
        )

        # write out
        with open(outfile, "a") as file:
            file.write(f"{_whichiter} {err:.6e}\n")
        if np.prod(np.shape(errvec)) > 1:
            with open(outfile + ".max", "a") as file:
                file.write(f"{_whichiter} {errmax:.6e}\n")
            with open(outfile + ".min", "a") as file:
                file.write(f"{_whichiter} {errmin:.6e}\n")
            with open(outfile + ".distribution", "a") as file:
                file.write(
                    f"{_whichiter}" + " ".join([f"{x:.6e}" for x in errvec.flatten()]) + "\n"
                )

        # print convergence message:
        if conv_bool:
            colorprefix = BOLD + GREEN
        elif (err < threshold) and (_gooditer <= Nsuccess):
            colorprefix = BOLD + YELLOW
        else:
            colorprefix = BOLD + RED

        if _whichiter < Nloop:
            if np.prod(np.shape(errvec)) > 1:
                print(colorprefix + "max error=" + COLOREND + f"{errmax:.6e}")
            print(
                colorprefix
                + "    " * (np.prod(np.shape(errvec)) > 1)
                + "error="
                + COLOREND
                + f"{err:.6e}"
            )
            if np.prod(np.shape(errvec)) > 1:
                print(colorprefix + "min error=" + COLOREND + f"{errmin:.6e}")
        else:
            if np.prod(np.shape(errvec)) > 1:
                print(colorprefix + "max error=" + COLOREND + f"{errmax:.6e}")
            print(
                colorprefix
                + "    " * (np.prod(np.shape(errvec)) > 1)
                + "error="
                + COLOREND
                + f"{err:.6e}"
            )
            if np.prod(np.shape(errvec)) > 1:
                print(colorprefix + "min error=" + COLOREND + f"{errmin:.6e}")
            print("Not converged after " + str(Nloop) + " iterations.")
            with open("ERROR.README", "a") as file:
                file.write("Not converged after " + str(Nloop) + " iterations.")
        print("\n")

    # pass to other cores:
    try:
        conv_bool = comm.bcast(conv_bool, root=0)
        err = comm.bcast(err, root=0)
        sys.stdout.flush()
    except:
        pass
    return err, conv_bool


import glob
import shutil


def copy_temp_folder(result_dir):
    """
    Finds and copies the temporary edipack folder from result_dir.
    Overwrites the previous backup.
    """
    # Find the temp folder matching "edipack-qi_*.tmp"
    temp_folders = glob.glob(os.path.join(result_dir, "edipack-*.tmp"))

    if not temp_folders:
        print("No temporary folder found.")
        return

    temp_folder = temp_folders[0]  # Pick the first match

    # Define the permanent backup folder
    backup_folder = os.path.join(result_dir, "edipack_data")

    # Remove the old backup if it exists
    if os.path.exists(backup_folder):
        print("Checkpoint: Delete Backup folder")
        shutil.rmtree(backup_folder)  # Delete the old backup

    # Copy the new temp folder
    shutil.copytree(temp_folder, backup_folder)

    print(f"Copied {temp_folder} -> {backup_folder} (overwriting previous backup).")


def plot_run_overview_AB(result_dir, Nbath, Norb):
    color = ["blue", "green", "red"]
    ms = 8  # marker size
    lw = 0.5  # line width

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except:
        rank = 0

    os.makedirs(f"{result_dir}/run_overviews", exist_ok=True)
    print("Plotting run overview.")

    obs_A = get_result_data(site="A")
    obs_B = get_result_data(site="B")
    num_iterations = obs_A["dens_1"].shape[0]
    iteration_axis = np.arange(num_iterations)

    fig, axes = plt.subplots(4, 4, figsize=(24, 16))

    # Plot mean densities
    for orb in range(Norb):
        axes[0, 0].plot(
            iteration_axis,
            obs_A[f"dens_{orb+1}"],
            linestyle=":",
            marker=".",
            color=color[orb],
            markersize=ms,
            linewidth=lw,
            alpha=1.0,
            label=f"dens_{orb}",
        )
        axes[0, 1].plot(
            iteration_axis,
            obs_B[f"dens_{orb+1}"],
            linestyle=":",
            marker=".",
            color=color[orb],
            markersize=ms,
            linewidth=lw,
            alpha=1.0,
            label=f"dens_{orb}",
        )

    axes[0, 0].set_title("Densities Site A")
    axes[0, 1].set_title("Densities Site B")

    for ax in [axes[0, 0], axes[0, 1]]:
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Density")
        ax.set_ylim(0.0, 2.0)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot double occupancies
    for orb in range(Norb):
        axes[0, 2].plot(
            iteration_axis,
            obs_A[f"docc_{orb+1}"],
            linestyle=":",
            marker=".",
            color=color[orb],
            markersize=ms,
            linewidth=lw,
            alpha=1.0,
            label=f"docc_{orb}",
        )
        axes[0, 3].plot(
            iteration_axis,
            obs_B[f"docc_{orb+1}"],
            linestyle=":",
            marker=".",
            color=color[orb],
            markersize=ms,
            linewidth=lw,
            alpha=1.0,
            label=f"docc_{orb}",
        )

    axes[0, 2].set_title("Double occupancies Site A")
    axes[0, 3].set_title("Double occupancies Site B")

    for ax in [axes[0, 2], axes[0, 3]]:
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Double Occupancies")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # plot bath fit parameters
    for orb in range(Norb):
        for bath_orb in range(Nbath):
            eps_list_A = obs_A["eps"][:, orb, bath_orb]
            eps_list_B = obs_B["eps"][:, orb, bath_orb]
            V_list_A = obs_A["V"][:, orb, bath_orb]
            V_list_B = obs_B["V"][:, orb, bath_orb]
            color = ["blue", "green", "red"]
            axes[1, 0].plot(
                iteration_axis,
                eps_list_A,
                marker=".",
                color=color[orb],
                linestyle=":",
                linewidth=lw,
                alpha=1.0,
                label=f"eps[{orb}{bath_orb}]",
            )
            axes[1, 1].plot(
                iteration_axis,
                eps_list_B,
                marker=".",
                color=color[orb],
                linestyle=":",
                linewidth=lw,
                alpha=1.0,
                label=f"eps[{orb}{bath_orb}]",
            )
            axes[1, 2].plot(
                iteration_axis,
                V_list_A,
                marker=".",
                color=color[orb],
                linestyle=":",
                linewidth=lw,
                alpha=1.0,
                label=f"V[{orb}{bath_orb}]",
            )
            axes[1, 3].plot(
                iteration_axis,
                V_list_B,
                marker=".",
                color=color[orb],
                linestyle=":",
                linewidth=lw,
                alpha=1.0,
                label=f"V[{orb}{bath_orb}]",
            )

    for ax in [axes[1, 0], axes[1, 1]]:
        ax.set_ylabel("bath energies")
        ax.set_xlabel("Iteration")
        ax.legend()

    for ax in [axes[1, 2], axes[1, 3]]:
        ax.set_ylabel("bath hybridizations")
        ax.set_xlabel("Iteration")
        ax.legend()

    # Plot energy
    axes[2, 0].plot(
        iteration_axis,
        obs_A["egs"],
        linestyle=":",
        marker=".",
        color="purple",
        markersize=ms,
        linewidth=lw,
        alpha=1.0,
        label="egs_A",
    )
    axes[2, 0].plot(
        iteration_axis,
        obs_B["egs"],
        linestyle=":",
        marker=".",
        color="pink",
        markersize=ms,
        linewidth=lw,
        alpha=1.0,
        label="egs_B",
    )

    axes[2, 0].set_title("Energy A and B site")
    axes[2, 0].set_xlabel("Iteration")
    axes[2, 0].set_ylabel("Energy")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Plot Z
    for orb in range(Norb):
        color = ["blue", "green", "red"]
        axes[2, 1].plot(
            iteration_axis,
            obs_A[f"Z_{orb+1}"],
            linestyle=":",
            marker=".",
            color=color[orb],
            markersize=ms,
            linewidth=lw,
            alpha=1.0,
            label=f"Z_{orb+1}",
        )
        axes[2, 2].plot(
            iteration_axis,
            obs_B[f"Z_{orb+1}"],
            linestyle=":",
            marker=".",
            color=color[orb],
            markersize=ms,
            linewidth=lw,
            alpha=1.0,
            label=f"Z_{orb+1}",
        )

    axes[2, 1].set_title("Z Site A")
    axes[2, 2].set_title("Z Site B")

    for ax in [axes[2, 1], axes[2, 2]]:
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Z")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot Error
    axes[2, 3].semilogy(
        iteration_axis,
        obs_A["err"],
        linestyle=":",
        marker=".",
        color="purple",
        markersize=ms,
        linewidth=lw,
        alpha=1.0,
        label="err",
    )

    axes[2, 3].set_title("error Site A")
    axes[2, 3].set_xlabel("Iteration")
    axes[2, 3].set_ylabel("error")
    axes[2, 3].legend()
    axes[2, 3].grid(True, alpha=0.3)

    # Plot self energy
    ar = HDFArchive("results.h5", "r")
    s_iw_A = ar["GFs"]["A_s_iw"]
    s_iw_B = ar["GFs"]["B_s_iw"]
    del ar

    for i in range(Norb):
        color = ["blue", "green", "red"]
        axes[3, 0].oplot(
            s_iw_A["up"][i, i].imag,
            name=f"S_{i}{i}",
            marker=".",
            color=color[i],
            linestyle=":",
            linewidth=lw,
            alpha=1.0,
            label=f"S_{i}{i}",
        )
        axes[3, 1].oplot(
            s_iw_B["up"][i, i].imag,
            name=f"S_{i}{i}",
            marker=".",
            color=color[i],
            linestyle=":",
            linewidth=lw,
            alpha=1.0,
            label=f"S_{i}{i}",
        )

    for ax in [axes[3, 0], axes[3, 1]]:
        ax.set_xlim(0.0, 2)
        ax.set_ylim(None, 0)
        ax.set_ylabel("Self energy")

    # Save plot
    fig.tight_layout()
    fig.savefig(f"{result_dir}/run_overviews/run_overview.pdf")
    plt.close(fig)


def plot_GFs_iteration(result_dir, iloop):
    os.makedirs(f"{result_dir}/GF_plots", exist_ok=True)
    print("Plotting Greens functions.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot self energy
    ar = HDFArchive("results.h5", "r")
    s_iw = ar["GFs"]["s_iw"]
    g_iw = ar["GFs"]["g_iw"]
    del ar

    # plot self energy
    for i in range(3):
        color = ["blue", "green", "red"]
        axes[0].oplot(
            s_iw["up"][i, i].imag,
            name=f"S_{i}{i}",
            marker=".",
            color=color[i],
            linestyle=":",
            linewidth=0.5,
            alpha=1.0,
            label=f"S_{i}{i}",
        )
    axes[0].set_xlim(0.0, 2)
    axes[0].set_ylim(None, 0)
    axes[0].set_ylabel("Self energy")

    # Plot Matsubara Greens function
    for i in range(3):
        color = ["blue", "green", "red"]
        axes[1].oplot(
            g_iw["up"][i, i].imag,
            name=f"G_{i}{i}",
            marker=".",
            color=color[i],
            linestyle=":",
            linewidth=0.5,
            alpha=1.0,
            label=f"G_{i}{i}",
        )
    axes[1].set_xlim(0.0, 2)
    axes[1].set_ylim(None, 0)
    axes[1].set_ylabel("Matsubara Greens function")

    # Save plot
    fig.tight_layout()
    fig.savefig(f"{result_dir}/GF_plots/GFs_overview_iloop={iloop}.pdf")
    plt.close(fig)


def plot_run_overview_A(result_dir, Nbath, Norb):
    color = ["blue", "green", "red"]
    ms = 8  # marker size
    lw = 0.5  # line width

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except:
        rank = 0

    os.makedirs(f"{result_dir}/run_overviews", exist_ok=True)
    print("Plotting run overview.")

    obs_A = get_result_data(site="A")
    num_iterations = obs_A["dens_1"].shape[0]
    iteration_axis = np.arange(num_iterations)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))

    # Plot mean densities
    ax = axes[0, 0]
    for orb in range(Norb):
        ax.plot(
            iteration_axis,
            obs_A[f"dens_{orb+1}"],
            linestyle=":",
            marker=".",
            color=color[orb],
            markersize=ms,
            linewidth=lw,
            alpha=1.0,
            label=f"dens_{orb}",
        )

    ax.set_title("Densities")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Density")
    ax.set_ylim(0.0, 2.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot double occupancies
    ax = axes[0, 1]
    for orb in range(Norb):
        ax.plot(
            iteration_axis,
            obs_A[f"docc_{orb+1}"],
            linestyle=":",
            marker=".",
            color=color[orb],
            markersize=ms,
            linewidth=lw,
            alpha=1.0,
            label=f"docc_{orb}",
        )

    ax.set_title("Double occupancies")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Double Occupancies")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # plot bath fit parameters
    ax1 = axes[0, 2]
    ax2 = axes[0, 3]
    for orb in range(Norb):
        for bath_orb in range(Nbath):
            eps_list_A = obs_A["eps"][:, orb, bath_orb]
            V_list_A = obs_A["V"][:, orb, bath_orb]
            color = ["blue", "green", "red"]
            ax1.plot(
                iteration_axis,
                eps_list_A,
                marker=".",
                color=color[orb],
                linestyle=":",
                linewidth=lw,
                alpha=1.0,
                label=f"eps[{orb}{bath_orb}]",
            )
            ax2.plot(
                iteration_axis,
                V_list_A,
                marker=".",
                color=color[orb],
                linestyle=":",
                linewidth=lw,
                alpha=1.0,
                label=f"V[{orb}{bath_orb}]",
            )

        ax1.set_ylabel("bath energies")
        ax1.set_xlabel("Iteration")
        ax1.legend()

        ax2.set_ylabel("bath hybridizations")
        ax2.set_xlabel("Iteration")
        ax2.legend()

    # Plot energy
    ax = axes[1, 0]
    ax.plot(
        iteration_axis,
        obs_A["egs"],
        linestyle=":",
        marker=".",
        color="purple",
        markersize=ms,
        linewidth=lw,
        alpha=1.0,
        label="egs_A",
    )

    ax.set_title("Energy")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot Z
    ax = axes[1, 1]
    for orb in range(Norb):
        color = ["blue", "green", "red"]
        ax.plot(
            iteration_axis,
            obs_A[f"Z_{orb+1}"],
            linestyle=":",
            marker=".",
            color=color[orb],
            markersize=ms,
            linewidth=lw,
            alpha=1.0,
            label=f"Z_{orb+1}",
        )

    ax.set_title("Z")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Z")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot Error
    ax = axes[1, 2]
    ax.semilogy(
        iteration_axis,
        obs_A["err"],
        linestyle=":",
        marker=".",
        color="purple",
        markersize=ms,
        linewidth=lw,
        alpha=1.0,
        label="err",
    )

    ax.set_title("error")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot self energy
    ax = axes[1, 3]
    ar = HDFArchive("results.h5", "r")
    s_iw_A = ar["GFs"]["A_s_iw"]
    del ar

    for i in range(Norb):
        color = ["blue", "green", "red"]
        ax.oplot(
            s_iw_A["up"][i, i].imag,
            name=f"S_{i}{i}",
            marker=".",
            color=color[i],
            linestyle=":",
            linewidth=lw,
            alpha=1.0,
            label=f"S_{i}{i}",
        )

    ax.set_xlim(0.0, 2)
    ax.set_ylim(None, 0)
    ax.set_ylabel("Self energy")

    # Save plot
    fig.tight_layout()
    fig.savefig(f"{result_dir}/run_overviews/run_overview.pdf")
    plt.close(fig)


def plot_GFs_iteration(result_dir, iloop):
    os.makedirs(f"{result_dir}/GF_plots", exist_ok=True)
    print("Plotting Greens functions.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot self energy
    ar = HDFArchive("results.h5", "r")
    s_iw = ar["GFs"]["s_iw"]
    g_iw = ar["GFs"]["g_iw"]
    del ar

    # plot self energy
    for i in range(3):
        color = ["blue", "green", "red"]
        axes[0].oplot(
            s_iw["up"][i, i].imag,
            name=f"S_{i}{i}",
            marker=".",
            color=color[i],
            linestyle=":",
            linewidth=0.5,
            alpha=1.0,
            label=f"S_{i}{i}",
        )
    axes[0].set_xlim(0.0, 2)
    axes[0].set_ylim(None, 0)
    axes[0].set_ylabel("Self energy")

    # Plot Matsubara Greens function
    for i in range(3):
        color = ["blue", "green", "red"]
        axes[1].oplot(
            g_iw["up"][i, i].imag,
            name=f"G_{i}{i}",
            marker=".",
            color=color[i],
            linestyle=":",
            linewidth=0.5,
            alpha=1.0,
            label=f"G_{i}{i}",
        )
    axes[1].set_xlim(0.0, 2)
    axes[1].set_ylim(None, 0)
    axes[1].set_ylabel("Matsubara Greens function")

    # Save plot
    fig.tight_layout()
    fig.savefig(f"{result_dir}/GF_plots/GFs_overview_iloop={iloop}.pdf")
    plt.close(fig)


def get_result_data(site="AB"):
    observable_names = [
        "dens_1",
        "dens_2",
        "dens_3",  # Single occupations
        "docc_1",
        "docc_2",
        "docc_3",  # double occupancies
        "nup_1",
        "nup_2",
        "nup_3",
        "ndw_1",
        "ndw_2",
        "ndw_3",
        "mag_1",
        "mag_2",
        "mag_3",
        "s2tot",
        "egs",
    ]

    Z_names = ["Z_1", "Z_2", "Z_3"]

    obs = {}

    assert site == "A" or site == "B"
    ar = HDFArchive("results.h5", "r")
    for observable in observable_names:
        obs[observable] = ar["observables"][f"{site}_{observable}"]
    for observable in Z_names:
        obs[observable] = ar["observables"][f"{site}_{observable}"]
    obs["err"] = ar["convergence"]["err"]
    obs["eps"] = ar["bath"][f"{site}_bath_eps"]
    obs["V"] = ar["bath"][f"{site}_bath_V"]

    obs["mean_docc_1_2"] = (obs["docc_1"] + obs["docc_2"]) / 2
    obs["mean_dens_1_2"] = (obs["dens_1"] + obs["dens_2"]) / 2
    del ar

    return obs


def save_observables(result_dir, site, Norb):
    assert site == "A" or site == "B"

    temp_folders = glob.glob(os.path.join(result_dir, "edipack-*.tmp"))

    if not temp_folders:
        print("No temporary folder found.")
        return

    temp_folder = temp_folders[0]  # Pick the first match

    observables_all = f"{temp_folder}/observables_all.ed"
    Z_all = f"{temp_folder}/Z_all.ed"

    with open(observables_all) as f:
        last_line = f.readlines()[-1]
    obs = pd.read_csv(StringIO(last_line), sep=r"\s+", header=None)

    column_names = [
        "dens_1",
        "dens_2",
        "dens_3",  # Single occupations
        "docc_1",
        "docc_2",
        "docc_3",  # double occupancies
        "nup_1",
        "nup_2",
        "nup_3",
        "ndw_1",
        "ndw_2",
        "ndw_3",
        "mag_1",
        "mag_2",
        "mag_3",
        "s2tot",
        "egs",
    ]

    obs.columns = column_names

    Z_column_names = ["Z_1", "Z_2", "Z_3"]
    with open(Z_all) as f:
        last_line = f.readlines()[-1]
    Z = pd.read_csv(StringIO(last_line), sep=r"\s+", header=None)
    Z.columns = Z_column_names

    # Save observables in hdf5 file
    ar = HDFArchive("results.h5", "a")
    obs_group = ar["observables"]
    if f"{site}_s2tot" not in obs_group:
        print("Creating observables")
        for orb in range(Norb):
            obs_group[f"{site}_dens_{orb+1}"] = np.array(obs[f"dens_{orb+1}"])
            obs_group[f"{site}_docc_{orb+1}"] = np.array(obs[f"docc_{orb+1}"])
            obs_group[f"{site}_nup_{orb+1}"] = np.array(obs[f"nup_{orb+1}"])
            obs_group[f"{site}_ndw_{orb+1}"] = np.array(obs[f"ndw_{orb+1}"])
            obs_group[f"{site}_mag_{orb+1}"] = np.array(obs[f"mag_{orb+1}"])
            obs_group[f"{site}_Z_{orb+1}"] = np.array(Z[f"Z_{orb+1}"])
        obs_group[f"{site}_s2tot"] = np.array(obs["s2tot"])
        obs_group[f"{site}_egs"] = np.array(obs["egs"])

    else:
        for orb in range(Norb):
            obs_group[f"{site}_dens_{orb+1}"] = np.concatenate(
                [obs_group[f"{site}_dens_{orb+1}"], obs[f"dens_{orb+1}"]], axis=0
            )
            obs_group[f"{site}_docc_{orb+1}"] = np.concatenate(
                [obs_group[f"{site}_docc_{orb+1}"], obs[f"docc_{orb+1}"]], axis=0
            )
            obs_group[f"{site}_nup_{orb+1}"] = np.concatenate(
                [obs_group[f"{site}_nup_{orb+1}"], obs[f"nup_{orb+1}"]], axis=0
            )
            obs_group[f"{site}_ndw_{orb+1}"] = np.concatenate(
                [obs_group[f"{site}_ndw_{orb+1}"], obs[f"ndw_{orb+1}"]], axis=0
            )
            obs_group[f"{site}_mag_{orb+1}"] = np.concatenate(
                [obs_group[f"{site}_mag_{orb+1}"], obs[f"mag_{orb+1}"]], axis=0
            )
            obs_group[f"{site}_Z_{orb+1}"] = np.concatenate(
                [obs_group[f"{site}_Z_{orb+1}"], Z[f"Z_{orb+1}"]], axis=0
            )
        obs_group[f"{site}_s2tot"] = np.concatenate(
            [obs_group[f"{site}_s2tot"], obs["s2tot"]], axis=0
        )
        obs_group[f"{site}_egs"] = np.concatenate([obs_group[f"{site}_egs"], obs["egs"]], axis=0)
