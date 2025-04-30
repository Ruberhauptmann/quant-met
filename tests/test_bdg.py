import numpy as np
from quant_met.bdg import bdg_hamiltonian


def test_bdg_shape_and_symmetry(square_lattice_tb):
    # Test point at Gamma
    k = np.array([0.0, 0.0, 0.0])

    delta_0 = 0.2

    bdg = bdg_hamiltonian(
        hamiltonian=square_lattice_tb,
        k=k,
        delta_orbital_basis=np.array([delta_0])
    )

    # Should return a 2x2 matrix for 1 orbital per site
    assert bdg.shape == (2, 2)

    assert np.all(bdg == np.array([[0.+0.j, 0.2-0.j], [0.2+0.j, -0.+0.j]]))
