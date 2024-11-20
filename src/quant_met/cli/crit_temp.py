# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Functions to run self-consistent calculation for the order parameter."""

import logging
from pathlib import Path

import h5py

from quant_met import mean_field
from quant_met.parameters import Parameters

from ._utils import _hamiltonian_factory

logger = logging.getLogger(__name__)


def crit_temp(parameters: Parameters) -> None:
    """Self-consistent calculation for the order parameter.

    Parameters
    ----------
    parameters: Parameters
        An instance of Parameters containing control settings, the model,
        and k-point specifications for the T_C calculation.
    """
    result_path = Path(parameters.control.outdir)
    result_path.mkdir(exist_ok=True, parents=True)

    h = _hamiltonian_factory(parameters=parameters.model, classname=parameters.model.name)

    delta_vs_temp, critical_temperatures, fit_fig = mean_field.search_crit_temp(
        h=h,
        k_space_grid=h.lattice.generate_bz_grid(
            ncols=parameters.k_points.nk1, nrows=parameters.k_points.nk2
        ),
        epsilon=parameters.control.conv_treshold,
        max_iter=parameters.control.max_iter,
        n_temp_points=parameters.control.n_temp_points,
    )

    logger.info("Search for T_C completed successfully.")
    logger.info("Obtained T_Cs: %s", critical_temperatures)

    fit_fig.savefig(
        result_path / f"{parameters.control.prefix}_critical_temperatures_fit.pdf",
        bbox_inches="tight",
    )

    result_file = result_path / f"{parameters.control.prefix}_critical_temperatures.hdf5"
    delta_vs_temp.to_hdf(result_file, key="delta_vs_temp")
    with h5py.File(result_file, mode="a") as file:
        for orbital, crit_temp_orbital in enumerate(critical_temperatures):
            file.attrs[f"T_C_{orbital}"] = crit_temp_orbital
    hamiltonian_file = result_path / f"{parameters.control.prefix}_sample_hamiltonian.hdf5"
    h.save(hamiltonian_file)

    logger.info("Results saved to %s", result_file)
