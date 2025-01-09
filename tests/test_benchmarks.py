# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT
import pytest
from quant_met import geometry, mean_field, parameters
import numpy as np


@pytest.mark.benchmark()
def test_benchmark_self_consistency_dressed_graphene() -> None:
    """Benchmark self-consistency for dressed Graphene."""
    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))

    dressed_graphene_h = mean_field.hamiltonians.DressedGraphene(
        parameters=parameters.DressedGrapheneParameters(
            hopping_gr=1,
            hopping_x=0.01,
            hopping_x_gr_a=1,
            lattice_constant=graphene_lattice.lattice_constant,
            chemical_potential=0,
            hubbard_int_orbital_basis=[1.0, 1.0, 1.0],
        )
    )

    mean_field.self_consistency_loop(
        h=dressed_graphene_h,
        k_space_grid=graphene_lattice.generate_bz_grid(40, 40),
        epsilon=1e-2,
        max_iter=1000,
    )
