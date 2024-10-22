# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Functions to run self-consistent calculation for the order parameter."""

from pathlib import Path

import numpy as np
from pydantic import BaseModel

from quant_met import mean_field
from quant_met.mean_field.hamiltonians import BaseHamiltonian
from quant_met.parameters import Parameters


def _hamiltonian_factory(classname: str, parameters: BaseModel) -> BaseHamiltonian:
    """Create a hamiltonian by its class name."""
    from quant_met.mean_field import hamiltonians

    cls = getattr(hamiltonians, classname)
    h: BaseHamiltonian = cls(parameters)
    return h


def scf(parameters: Parameters) -> None:
    """Self-consistent calculation for the order parameter."""
    result_path = Path(parameters.control.outdir)
    result_path.mkdir(exist_ok=True, parents=True)
    h = _hamiltonian_factory(parameters=parameters.model, classname=parameters.model.name)
    solved_h = mean_field.self_consistency_loop(
        h=h,
        k_space_grid=h.lattice.generate_bz_grid(
            ncols=parameters.k_points.nk1, nrows=parameters.k_points.nk2
        ),
        beta=np.float64(parameters.control.beta),
        epsilon=parameters.control.conv_treshold,
        q=parameters.control.q,
    )
    print(solved_h.delta_orbital_basis)
    result_file = result_path / f"{parameters.control.prefix}.hdf5"
    solved_h.save(filename=result_file)
