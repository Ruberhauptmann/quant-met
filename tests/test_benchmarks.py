# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT
from quant_met import mean_field, parameters
import pytest
import numpy as np


@pytest.mark.slow_integration_test
def test_benchmark_self_consistency_one_band(benchmark) -> None:
    """Benchmark self-consistency for the one band model."""
    one_band_h = mean_field.hamiltonians.OneBand(
        parameters=parameters.OneBandParameters(
            hopping=1,
            lattice_constant=1,
            chemical_potential=0,
            hubbard_int_orbital_basis=[1.0],
        )
    )
    k_space_grid = one_band_h.lattice.generate_bz_grid(ncols=30, nrows=30)

    benchmark(lambda: mean_field.self_consistency_loop(
        h=one_band_h,
        k_space_grid=k_space_grid,
        delta_init=np.array([1], dtype=np.complex64),
        epsilon=1e-2,
    ))


@pytest.mark.slow_integration_test
def test_benchmark_self_consistency_two_band(benchmark) -> None:
    """Benchmark self-consistency for the two band model."""
    one_band_h = mean_field.hamiltonians.TwoBand(
        parameters=parameters.TwoBandParameters(
            hopping=1,
            lattice_constant=1,
            chemical_potential=0,
            hubbard_int_orbital_basis=[1.0, 1.0],
        )
    )
    k_space_grid = one_band_h.lattice.generate_bz_grid(ncols=30, nrows=30)

    benchmark(lambda: mean_field.self_consistency_loop(
        h=one_band_h,
        k_space_grid=k_space_grid,
        delta_init=np.array([1, 1], dtype=np.complex64),
        epsilon=1e-2,
    ))
