# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT
from quant_met import mean_field, parameters


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

    benchmark(mean_field.self_consistency_loop(
        h=one_band_h,
        k_space_grid=k_space_grid,
        epsilon=1e-2,
    ))
