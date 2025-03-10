# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Test random sample Hamiltonians."""

import numpy as np
import numpy.typing as npt
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import (builds, floats, from_type, integers, just,
                                   one_of, register_type_strategy, tuples)
from pydantic import BaseModel
from scipy import linalg

from quant_met import parameters
from quant_met.mean_field.hamiltonians import BaseHamiltonian

register_type_strategy(
    parameters.OneBandParameters,
    builds(
        parameters.OneBandParameters,
        lattice_constant=floats(
            min_value=0,
            max_value=1e4,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        hopping=floats(
            min_value=0,
            max_value=1e6,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        chemical_potential=floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        hubbard_int_orbital_basis=arrays(
            shape=(1,),
            dtype=float,
            elements=floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
        ),
    ),
)

register_type_strategy(
    parameters.GrapheneParameters,
    builds(
        parameters.GrapheneParameters,
        lattice_constant=floats(
            min_value=0,
            max_value=1e4,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        hopping=floats(
            min_value=0,
            max_value=1e6,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        chemical_potential=floats(max_value=1e6, allow_nan=False, allow_infinity=False),
        hubbard_int_orbital_basis=arrays(
            shape=(2,),
            dtype=float,
            elements=floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
        ),
    ),
)

register_type_strategy(
    parameters.DressedGrapheneParameters,
    builds(
        parameters.DressedGrapheneParameters,
        lattice_constant=floats(
            min_value=0,
            max_value=1e5,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        hopping_gr=floats(
            min_value=0,
            max_value=1e6,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        hopping_x=floats(
            min_value=0,
            max_value=1e6,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        chemical_potential=floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        hopping_x_gr_a=floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
        hubbard_int_orbital_basis=arrays(
            shape=(3,),
            dtype=float,
            elements=floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
        ),
    ),
)


def _hamiltonian_factory(classname: str, input_parameters: BaseModel) -> BaseHamiltonian:
    """Create a hamiltonian by its class name."""
    from quant_met.mean_field import hamiltonians

    cls = getattr(hamiltonians, classname)
    h: BaseHamiltonian = cls(input_parameters)
    return h


@given(
    sample_parameters=one_of(
        from_type(parameters.DressedGrapheneParameters),
        from_type(parameters.GrapheneParameters),
        from_type(parameters.OneBandParameters),
    ),
    k=arrays(
        shape=tuples(integers(min_value=0, max_value=100), just(2)),
        dtype=float,
        elements=floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
)
def test_hamiltonians(sample_parameters: BaseModel, k: npt.NDArray) -> None:
    """Test Hamiltonians with random parameters."""
    sample = _hamiltonian_factory(
        input_parameters=sample_parameters, classname=sample_parameters.name
    )

    assert sample.name == sample_parameters.name

    sample.delta_orbital_basis = np.array([0 for _ in range(sample.number_of_bands)])

    bdg_energies = sample.diagonalize_bdg(k=k)[0].flatten()

    nonint_energies = np.array(
        [[+E, -E] for E in sample.diagonalize_nonint(k=k)[0].flatten()]
    ).flatten()
    assert np.allclose(
        np.sort(np.nan_to_num(bdg_energies.flatten())),
        np.sort(np.nan_to_num(nonint_energies)),
    )

    h_k_space = sample.hamiltonian(k)
    if h_k_space.ndim == 2:
        h_k_space = np.expand_dims(h_k_space, axis=0)

    assert len(sample.hubbard_int_orbital_basis) == sample.number_of_bands
    for h in h_k_space:
        assert h.shape[0] == sample.number_of_bands
        assert h.shape[1] == sample.number_of_bands
        assert linalg.ishermitian(h)
