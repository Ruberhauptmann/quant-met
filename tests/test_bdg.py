import numpy as np
import sisl

from quant_met.bdg import diagonalize_bdg, bdg_hamiltonian, calculate_current_density


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


def test_sc_current(square_lattice_tb):
    k_grid = sisl.MonkhorstPack(square_lattice_tb.geometry, [10, 10, 1])

    delta_0 = 0.2

    bdg_energies, bdg_wavefunctions = diagonalize_bdg(
        hamiltonian=square_lattice_tb,
        kgrid=k_grid,
        delta_orbital_basis=np.array([delta_0]),
        q=np.array([0.0, 0.0, 0.0])
    )
    current = calculate_current_density(
        hamiltonian=square_lattice_tb,
        k=k_grid,
        bdg_energies=bdg_energies,
        bdg_wavefunctions=bdg_wavefunctions,
        beta=1000
    )

    assert np.all(np.equal(current, np.array([0.0, 0.0])))
