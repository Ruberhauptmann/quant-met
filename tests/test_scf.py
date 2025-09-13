import numpy as np
from quant_met.routines import self_consistency_loop
import sisl


def test_scf(square_lattice_tb):
    k_grid = sisl.MonkhorstPack(square_lattice_tb.geometry, [10, 10, 1])

    solved_gap = self_consistency_loop(
        hamiltonian=square_lattice_tb,
        kgrid=k_grid,
        beta=np.inf,
        hubbard_int_orbital_basis=np.array([0.0]),
        epsilon=1e-2,
        q=np.array([0.0, 0.0, 0.0])
    )
    assert np.allclose(solved_gap, 0)
    solved_gap = self_consistency_loop(
        hamiltonian=square_lattice_tb,
        kgrid=k_grid,
        beta=np.inf,
        hubbard_int_orbital_basis=np.array([0.0]),
        epsilon=1e-2,
        delta_init=np.array([1.0]),
        q=np.array([0.0, 0.0, 0.0])
    )
    assert np.allclose(solved_gap, 0)


def test_scf_max_iterations(square_lattice_tb):
    k_grid = sisl.MonkhorstPack(square_lattice_tb.geometry, [10, 10, 1])

    solved_gap = self_consistency_loop(
        hamiltonian=square_lattice_tb,
        kgrid=k_grid,
        beta=np.inf,
        hubbard_int_orbital_basis=np.array([0.0]),
        epsilon=1e-2,
        q=np.array([0.0, 0.0, 0.0])
    )
    assert np.allclose(solved_gap, 0)
