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

    h = _hamiltonian_factory(parameters=parameters.model, classname=parameters.model.name)
    k_space_grid = h.lattice.generate_bz_grid(
        ncols=parameters.k_points.nk1, nrows=parameters.k_points.nk2
    )

    solved_h = mean_field.self_consistency_loop(
        h=h,
        k_space_grid=k_space_grid,
        epsilon=parameters.control.conv_treshold,
        max_iter=parameters.control.max_iter,
    )

    logger.info("Self-consistency loop completed successfully.")
    logger.debug("Obtained delta values: %s", solved_h.delta_orbital_basis)

    result_file = result_path / f"{parameters.control.prefix}.hdf5"
    solved_h.save(filename=result_file)
    logger.info("Results saved to %s", result_file)

    if parameters.control.calculate_additional is True:
        logger.info("Calculating additional things.")
        current = solved_h.calculate_current_density(k=k_space_grid)
        free_energy = solved_h.calculate_free_energy(k=k_space_grid)
        sf_weight_conv, sf_weight_geom = solved_h.calculate_superfluid_weight(k=k_space_grid)

        with h5py.File(result_file, "a") as f:
            f.attrs["current_x"] = current[0]
            f.attrs["current_y"] = current[1]
            f.attrs["free_energy"] = free_energy
            f.attrs["sf_weight_conv_xx"] = sf_weight_conv[0, 0]
            f.attrs["sf_weight_conv_xy"] = sf_weight_conv[0, 1]
            f.attrs["sf_weight_conv_yx"] = sf_weight_conv[1, 0]
            f.attrs["sf_weight_conv_yy"] = sf_weight_conv[1, 1]
            f.attrs["sf_weight_geom_xx"] = sf_weight_geom[0, 0]
            f.attrs["sf_weight_geom_xy"] = sf_weight_geom[0, 1]
            f.attrs["sf_weight_geom_yx"] = sf_weight_geom[1, 0]
            f.attrs["sf_weight_geom_yy"] = sf_weight_geom[1, 1]

        logger.info("Additional results saved to %s", result_file)
