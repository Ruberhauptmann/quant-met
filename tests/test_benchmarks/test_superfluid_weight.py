# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT
import numpy as np
from quant_met import mean_field, parameters
import pytest


@pytest.mark.slow_integration_test
def test_benchmark_superfluid_weight_two_band(benchmark) -> None:
    """Benchmark superfluid weight for the dressed graphene model."""
    one_band_h = mean_field.hamiltonians.TwoBand(
        parameters=parameters.TwoBandParameters(
            hopping=1,
            lattice_constant=1,
            chemical_potential=0,
            hubbard_int_orbital_basis=[1.0, 1.0],
            delta=np.array([1.0, 1.0], dtype=np.complex128)
        )
    )
    k_space_grid = one_band_h.lattice.generate_bz_grid(ncols=10, nrows=10)

    benchmark(lambda: mean_field.superfluid_weight(
        h=one_band_h,
        k=k_space_grid)
    )
