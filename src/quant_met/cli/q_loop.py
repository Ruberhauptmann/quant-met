"""Functions to run self-consistent calculation for the order parameter."""

import logging
from pathlib import Path

import h5py
import numpy as np
import sisl

from quant_met import routines
from quant_met.parameters import Parameters
from quant_met.parameters.control import CritTemp, QLoop

logger = logging.getLogger(__name__)


def q_loop(parameters: Parameters) -> None:
    """Self-consistent calculation for the order parameter.

    Parameters
    ----------
    parameters: Parameters
        An instance of Parameters containing control settings, the model,
        and k-point specifications for the T_C calculation.
    """
    if not isinstance(parameters.control, QLoop):
        raise ValueError("Wrong parameters for q-loop.")

    result_path = Path(parameters.control.outdir)
    result_path.mkdir(exist_ok=True, parents=True)

    hamiltonian = sisl.get_sile(parameters.control.hamiltonian_file).read_hamiltonian()
    k_grid_obj = sisl.MonkhorstPack(
        hamiltonian.geometry, [parameters.k_points.nk1, parameters.k_points.nk2, 1]
    )

    if isinstance(parameters.control.crit_temp, CritTemp):
        delta_vs_temp, critical_temperatures, fit_fig = routines.search_crit_temp(
            hamiltonian=hamiltonian,
            kgrid=k_grid_obj,
            hubbard_int_orbital_basis=np.array(parameters.control.hubbard_int_orbital_basis),
            epsilon=parameters.control.conv_treshold,
            max_iter=parameters.control.max_iter,
            n_temp_points=20,
        )
        logger.info("Search for T_C completed successfully.")
        logger.info("Obtained T_Cs: %s", critical_temperatures)

        fit_fig.savefig(
            result_path / f"{parameters.control.prefix}_critical_temperatures_fit.pdf",
            bbox_inches="tight",
        )

        result_file_crit_temp = (
            result_path / f"{parameters.control.prefix}_critical_temperatures.hdf5"
        )
        delta_vs_temp.to_hdf(result_file_crit_temp, key="delta_vs_temp")
        with h5py.File(result_file_crit_temp, mode="a") as file:
            for orbital, crit_temp_orbital in enumerate(critical_temperatures):
                file.attrs[f"T_C_{orbital}"] = crit_temp_orbital

        logger.info("Results saved to %s", result_file_crit_temp)
    else:
        critical_temperatures = []
        with h5py.File(f"{parameters.control.crit_temp}", mode="r") as file:
            for key, critical_temperature in file.attrs.items():
                if key.startswith("T_C"):
                    critical_temperatures.append(critical_temperature)
        logger.info("Read critical temperatures from file.")
        logger.info("Obtained T_Cs: %s", critical_temperatures)

    delta_vs_q, fig = routines.loop_over_q(
        hamiltonian=hamiltonian,
        kgrid=k_grid_obj,
        hubbard_int_orbital_basis=np.array(parameters.control.hubbard_int_orbital_basis),
        epsilon=parameters.control.conv_treshold,
        max_iter=parameters.control.max_iter,
        n_q_points=parameters.control.n_q_points,
        crit_temps=np.array(critical_temperatures),
    )

    result_file_q = result_path / f"{parameters.control.prefix}_q.hdf5"
    for key, df in delta_vs_q.items():
        df.to_hdf(result_file_q, key=f"temp_{float(key):.4f}")

    fig.savefig(f"{parameters.control.outdir}/{parameters.control.prefix}_q_plot.pdf")
