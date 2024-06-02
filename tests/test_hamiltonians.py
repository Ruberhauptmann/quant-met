import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import (
    builds,
    floats,
    from_type,
    one_of,
    register_type_strategy,
)
from scipy import linalg

from quant_met import hamiltonians

register_type_strategy(
    hamiltonians.GrapheneHamiltonian,
    builds(
        hamiltonians.GrapheneHamiltonian,
        a=floats(min_value=0, exclude_min=True, allow_nan=False, allow_infinity=False),
        t_nn=floats(allow_nan=False, allow_infinity=False),
        mu=floats(allow_nan=False, allow_infinity=False),
        coulomb_gr=floats(allow_nan=False, allow_infinity=False),
    ),
)

register_type_strategy(
    hamiltonians.EGXHamiltonian,
    builds(
        hamiltonians.EGXHamiltonian,
        a=floats(min_value=0, exclude_min=True, allow_nan=False, allow_infinity=False),
        t_gr=floats(allow_nan=False, allow_infinity=False),
        t_x=floats(allow_nan=False, allow_infinity=False),
        mu=floats(allow_nan=False, allow_infinity=False),
        V=floats(allow_nan=False, allow_infinity=False),
        U_gr=floats(allow_nan=False, allow_infinity=False),
        U_x=floats(allow_nan=False, allow_infinity=False),
    ),
)


@given(
    sample=one_of(
        from_type(hamiltonians.GrapheneHamiltonian),
        from_type(hamiltonians.EGXHamiltonian),
    ),
    k=arrays(
        shape=(2,),
        dtype=float,
        elements=floats(allow_nan=False, allow_infinity=False),
    ),
)
def test_samples(sample: hamiltonians.BaseHamiltonian, k: npt.NDArray):
    assert linalg.ishermitian(sample.hamiltonian_k_space(k)[0])
