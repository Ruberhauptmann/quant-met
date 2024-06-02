import numpy as np
import numpy.typing as npt
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import builds, floats, from_type, register_type_strategy
from scipy import linalg

from quant_met import hamiltonians


class TestGraphene:
    register_type_strategy(
        hamiltonians.GrapheneHamiltonian,
        builds(
            hamiltonians.GrapheneHamiltonian,
            a=floats(
                min_value=0, exclude_min=True, allow_nan=False, allow_infinity=False
            ),
            t_nn=floats(allow_nan=False, allow_infinity=False),
            mu=floats(allow_nan=False, allow_infinity=False),
            coulomb_gr=floats(allow_nan=False, allow_infinity=False),
        ),
    )

    @given(
        sample=from_type(hamiltonians.GrapheneHamiltonian),
        k=arrays(
            shape=(2,),
            dtype=float,
            elements=floats(allow_nan=False, allow_infinity=False),
        ),
    )
    def test_samples(self, sample: hamiltonians.GrapheneHamiltonian, k: npt.NDArray):
        assert linalg.ishermitian(sample.hamiltonian_k_space(k)[0])
