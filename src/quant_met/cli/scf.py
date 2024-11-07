# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Functions to run self-consistent calculation for the order parameter."""

import logging
from pathlib import Path

from quant_met import mean_field
from quant_met.mean_field.hamiltonians import BaseHamiltonian
from quant_met.parameters import HamiltonianParameters, Parameters

logger = logging.getLogger(__name__)


def _hamiltonian_factory(
    classname: str, parameters: HamiltonianParameters
) -> BaseHamiltonian[HamiltonianParameters]:
    """Create a Hamiltonian by its class name.

    Parameters
    ----------
    classname: str
        The name of the Hamiltonian class to instantiate.
    parameters: HamiltonianParameters
        An instance of HamiltonianParameters containing all necessary
        configuration for the specific Hamiltonian.

    Returns
    -------
    BaseHamiltonian[HamiltonianParameters]
        An instance of the specified Hamiltonian class.
    """
    from quant_met.mean_field import hamiltonians

    cls = getattr(hamiltonians, classname)
    h: BaseHamiltonian[HamiltonianParameters] = cls(parameters)
    return h


def scf(parameters: Parameters) -> None:
    """Self-consistent calculation for the order parameter.

    Parameters
    ----------
    parameters: Parameters
        An instance of Parameters containing control settings, the model,
        and k-point specifications for the self-consistency calculation.
    """
    result_path = Path(parameters.control.outdir)
    result_path.mkdir(exist_ok=True, parents=True)

    logger.info("Initializing Hamiltonian factory.")
    h = _hamiltonian_factory(parameters=parameters.model, classname=parameters.model.name)

    logger.info("Starting self-consistency loop.")
    solved_h = mean_field.self_consistency_loop(
        h=h,
        k_space_grid=h.lattice.generate_bz_grid(
            ncols=parameters.k_points.nk1, nrows=parameters.k_points.nk2
        ),
        epsilon=parameters.control.conv_treshold,
    )

    logger.info("Self-consistency loop completed successfully.")
    logger.debug("Obtained delta values: %s", solved_h.delta_orbital_basis)

    result_file = result_path / f"{parameters.control.prefix}.hdf5"
    solved_h.save(filename=result_file)
    logger.info("Results saved to %s", result_file)
