import numpy as np

from quant_met.routines import search_crit_temp
import sisl


def test_scf(square_lattice_tb):
    k_grid = sisl.MonkhorstPack(square_lattice_tb.geometry, [10, 10, 1])

    delta_vs_temp, critical_temp_list, fit_fig = search_crit_temp(
        hamiltonian=square_lattice_tb,
        kgrid=k_grid,
        hubbard_int_orbital_basis=np.array([1.0]),
        n_temp_points=15,
        max_iter=500,
        epsilon=1e-1,
        q=np.array([0.0, 0.0, 0.0]),
        beta_init=10
    )
    assert np.allclose(np.array(critical_temp_list), 0.25, rtol=0.1, atol=0.05)
