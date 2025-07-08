"""Functions to run self-consistent calculation for the order parameter."""

import logging
from pathlib import Path

import h5py
import sisl

from quant_met import bdg, routines
from quant_met.parameters import Parameters

logger = logging.getLogger(__name__)


def scf(parameters: Parameters) -> None:
    """Self-consistent calculation for the order parameter.

    Parameters
    ----------
    parameters: Parameters
        An instance of Parameters containing control settings.
    """
    result_path = Path(parameters.control.outdir)
    result_path.mkdir(exist_ok=True, parents=True)
    result_file = result_path / f"{parameters.control.prefix}.hdf5"

    hamiltonian = sisl.get_sile(parameters.control.hamiltonian_file).read_hamiltonian()
    k_grid_obj = sisl.MonkhorstPack(
        hamiltonian.geometry, [parameters.k_points.nk1, parameters.k_points.nk2, 1]
    )

    solved_gap = routines.self_consistency_loop(
        hamiltonian=hamiltonian,
        kgrid=k_grid_obj,
        beta=parameters.control.beta,
        hubbard_int_orbital_basis=parameters.control.hubbard_int_orbital_basis,
        epsilon=parameters.control.conv_treshold,
        max_iter=parameters.control.max_iter,
        q=parameters.control.q,
    )

    logger.info("Self-consistency loop completed successfully.")
    logger.debug("Obtained delta values: %s", solved_gap)

    with h5py.File(result_file, "a") as f:
        f.create_dataset("delta", data=solved_gap)
    logger.info("Results saved to %s", result_file)

    if parameters.control.calculate_additional is True:
        logger.info("Calculating additional things.")

        bdg_energies, bdg_wavefunctions = bdg.diagonalize_bdg(
            hamiltonian=hamiltonian,
            kgrid=k_grid_obj,
            delta_orbital_basis=solved_gap,
            q=parameters.control.q,
        )

        current = bdg.calculate_current_density(
            hamiltonian=hamiltonian,
            k=k_grid_obj,
            bdg_energies=bdg_energies,
            bdg_wavefunctions=bdg_wavefunctions,
            beta=parameters.control.beta,
        )
        sf_weight_conv, sf_weight_geom = bdg.calculate_superfluid_weight(
            hamiltonian=hamiltonian,
            kgrid=k_grid_obj,
            beta=parameters.control.beta,
            delta_orbital_basis=solved_gap,
        )

        with h5py.File(result_file, "a") as f:
            f.attrs["current_x"] = current[0]
            f.attrs["current_y"] = current[1]
            f.attrs["sf_weight_conv_xx"] = sf_weight_conv[0, 0]
            f.attrs["sf_weight_conv_xy"] = sf_weight_conv[0, 1]
            f.attrs["sf_weight_conv_yx"] = sf_weight_conv[1, 0]
            f.attrs["sf_weight_conv_yy"] = sf_weight_conv[1, 1]
            f.attrs["sf_weight_geom_xx"] = sf_weight_geom[0, 0]
            f.attrs["sf_weight_geom_xy"] = sf_weight_geom[0, 1]
            f.attrs["sf_weight_geom_yx"] = sf_weight_geom[1, 0]
            f.attrs["sf_weight_geom_yy"] = sf_weight_geom[1, 1]

        logger.info("Additional results saved to %s", result_file)
