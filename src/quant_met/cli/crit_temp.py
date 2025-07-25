"""Functions to run self-consistent calculation for the order parameter."""

import logging
from pathlib import Path

import h5py
import numpy as np
import sisl

from quant_met import routines
from quant_met.parameters import Parameters
from quant_met.parameters.control import CritTemp

logger = logging.getLogger(__name__)


def crit_temp(parameters: Parameters) -> None:
    """Self-consistent calculation for the order parameter.

    Parameters
    ----------
    parameters: Parameters
        An instance of Parameters containing control settings, the model,
        and k-point specifications for the T_C calculation.
    """
    if not isinstance(parameters.control, CritTemp):
        err_msg = "Wrong parameters for crit-temp."
        raise TypeError(err_msg)

    result_path = Path(parameters.control.outdir)
    result_path.mkdir(exist_ok=True, parents=True)

    hamiltonian = sisl.get_sile(parameters.control.hamiltonian_file).read_hamiltonian()
    k_grid_obj = sisl.MonkhorstPack(
        hamiltonian.geometry,
        [parameters.k_points.nk1, parameters.k_points.nk2, 1],
    )

    delta_vs_temp, critical_temperatures, fit_fig = routines.search_crit_temp(
        hamiltonian=hamiltonian,
        kgrid=k_grid_obj,
        hubbard_int_orbital_basis=np.array(parameters.control.hubbard_int_orbital_basis),
        epsilon=parameters.control.conv_treshold,
        max_iter=parameters.control.max_iter,
        q=np.array(parameters.control.q),
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

    logger.info("Results saved to %s", result_file)
