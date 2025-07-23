"""Functions to run self-consistent calculation for the order parameter."""

import logging
from pathlib import Path

import h5py
import pandas as pd
import sisl

from quant_met import routines
from quant_met.parameters import Parameters
from quant_met.parameters.control import QAnalysis

logger = logging.getLogger(__name__)


def q_analysis(parameters: Parameters) -> None:
    """Self-consistent calculation for the order parameter.

    Parameters
    ----------
    parameters: Parameters
        An instance of Parameters containing control settings, the model,
        and k-point specifications for the T_C calculation.
    """
    if not isinstance(parameters.control, QAnalysis):
        raise ValueError("Wrong parameters for q-loop.")

    q_data = {}
    with h5py.File(f"{parameters.control.q_data}") as f:
        for key in f.keys():
            q_data.update({key: None})

    for key in q_data:
        q_data[key] = pd.read_hdf(f"{parameters.control.q_data}", key=key)

    hamiltonian = sisl.get_sile(parameters.control.hamiltonian_file).read_hamiltonian()

    (
        lengths_vs_temp,
        gap_and_current_fig,
    ) = routines.get_lengths_vs_temp(q_data=q_data, hamiltonian=hamiltonian)

    result_file = Path(f"{parameters.control.outdir}/{parameters.control.prefix}_sc_lengths.hdf5")
    result_file.unlink()
    lengths_vs_temp.to_hdf(result_file, key="lengths_vs_temp")
    gap_and_current_fig.savefig(
        f"{parameters.control.outdir}/{parameters.control.prefix}_gap_and_current_vs_q.pdf"
    )

    zero_temp_lengths, length_vs_temp_fig = routines.get_zero_temperature_values(
        hamiltonian=hamiltonian, lengths_vs_temp=lengths_vs_temp
    )
    zero_temp_lengths.to_hdf(result_file, key="zero_temp_lengths")
    length_vs_temp_fig.savefig(
        f"{parameters.control.outdir}/{parameters.control.prefix}_lengths_vs_temperature.pdf"
    )
