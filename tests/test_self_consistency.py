# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

from quant_met import geometry, mean_field
import numpy as np

def test_self_consistency():
    graphene_lattice = geometry.Graphene(lattice_constant=np.sqrt(3))

    egx_h = mean_field.EGXHamiltonian(
        hopping_gr=1,
        hopping_x=0.01,
        hopping_x_gr_a=1,
        lattice_constant=graphene_lattice.lattice_constant,
        chemical_potential=0,
        hubbard_int_gr=0,
        hubbard_int_x=0,
    )
    assert np.allclose(
        mean_field.self_consistency_loop(h=egx_h, number_of_k_points=50, epsilon=1e-4, beta=100).delta_orbital_basis, np.zeros(3)
    )
    assert np.allclose(
        mean_field.self_consistency_loop(h=egx_h, number_of_k_points=50, epsilon=1e-4, beta=100, q=np.array([0, 0])).delta_orbital_basis, np.zeros(3)
    )
