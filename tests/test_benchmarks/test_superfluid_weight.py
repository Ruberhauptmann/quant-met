# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT
from quant_met import mean_field, parameters
import pytest


@pytest.mark.slow_integration_test
def test_benchmark_superfluid_weight_one_band(benchmark) -> None:
    """Benchmark superfluid weight for the one band model."""
    one_band_h = mean_field.hamiltonians.OneBand(
        parameters=parameters.OneBandParameters(
            hopping=1,
            lattice_constant=1,
            chemical_potential=0,
            hubbard_int_orbital_basis=[1.0],
            delta=[1.0]
        )
    )
    k_space_grid = one_band_h.lattice.generate_bz_grid(ncols=30, nrows=30)

    benchmark(lambda: mean_field.superfluid_weight(
        h=one_band_h,
        k=k_space_grid)
    )


@pytest.mark.slow_integration_test
def test_benchmark_superfluid_weight_two_band(benchmark) -> None:
    """Benchmark superfluid weight for the two band model."""
    two_band_h = mean_field.hamiltonians.TwoBand(
        parameters=parameters.TwoBandParameters(
            hopping=1,
            lattice_constant=1,
            chemical_potential=0,
            hubbard_int_orbital_basis=[1.0, 1.0],
            delta=[1.0, 1.0]
        )
    )
    k_space_grid = two_band_h.lattice.generate_bz_grid(ncols=30, nrows=30)

    benchmark(lambda: mean_field.superfluid_weight(
        h=two_band_h,
        k=k_space_grid)
    )


@pytest.mark.slow_integration_test
def test_benchmark_superfluid_weight_three_band(benchmark) -> None:
    """Benchmark superfluid weight for the three band model."""
    three_band_h = mean_field.hamiltonians.ThreeBand(
        parameters=parameters.ThreeBandParameters(
            hopping=1,
            lattice_constant=1,
            chemical_potential=0,
            hubbard_int_orbital_basis=[1.0, 1.0, 1.0],
            delta=[1.0, 1.0, 1.0]
        )
    )
    k_space_grid = three_band_h.lattice.generate_bz_grid(ncols=30, nrows=30)

    benchmark(lambda: mean_field.superfluid_weight(
        h=three_band_h,
        k=k_space_grid)
    )


@pytest.mark.slow_integration_test
def test_benchmark_superfluid_weight_dressed_graphene(benchmark) -> None:
    """Benchmark superfluid weight for the dressed graphene model."""
    dressed_graphene_h = mean_field.hamiltonians.DressedGraphene(
        parameters=parameters.DressedGrapheneParameters(
            hopping_gr=1,
            hopping_x=0.01,
            hopping_x_gr_a=1,
            lattice_constant=1,
            chemical_potential=0,
            hubbard_int_orbital_basis=[1.0, 1.0, 1.0],
            delta=[1.0, 1.0, 1.0]
        )
    )
    k_space_grid = dressed_graphene_h.lattice.generate_bz_grid(ncols=30, nrows=30)

    benchmark(lambda: mean_field.superfluid_weight(
        h=dressed_graphene_h,
        k=k_space_grid)
    )
