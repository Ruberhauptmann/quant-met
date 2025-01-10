# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT
import numpy as np
from quant_met import mean_field, parameters
import pytest


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
            delta=np.array([1.0, 1.0, 1.0], dtype=np.complex128)
        )
    )
    k_space_grid = dressed_graphene_h.lattice.generate_bz_grid(ncols=30, nrows=30)

    benchmark(lambda: mean_field.superfluid_weight(
        h=dressed_graphene_h,
        k=k_space_grid)
    )
