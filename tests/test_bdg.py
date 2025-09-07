import numpy as np
from quant_met.bdg import bdg_hamiltonian


def test_bdg_square_lattice(square_lattice_tb):
    k = np.array([0.0, 0.0, 0.0])

    delta_0 = 0.2

    bdg = bdg_hamiltonian(
        hamiltonian=square_lattice_tb,
        k=k,
        delta_orbital_basis=np.array([delta_0]),
        q=np.array([0.0, 0.0, 0.0])
    )

    assert bdg.shape == (2, 2)

    assert np.all(bdg == np.array([[0.+0.j, 0.2-0.j], [0.2+0.j, -0.+0.j]]))
