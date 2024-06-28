import numpy as np

from quant_met import hamiltonians, utils


def test_free_energy(ndarrays_regression):
    t_gr = 1
    t_x = 0.01
    V = 1
    mu = 1
    lattice_constant = np.sqrt(3)
    all_K_points = (
        4
        * np.pi
        / (3 * lattice_constant)
        * np.array(
            [
                (np.sin(i * np.pi / 6), np.cos(i * np.pi / 6))
                for i in [1, 3, 5, 7, 9, 11]
            ]
        )
    )
    egx_h = hamiltonians.EGXHamiltonian(
        t_gr=t_gr, t_x=t_x, V=V, a=lattice_constant, mu=mu, U_gr=1, U_x=1
    )

    BZ_grid = utils.generate_uniform_grid(
        10, 10, all_K_points[1], all_K_points[5], origin=np.array([0, 0])
    )

    free_energy = hamiltonians.free_energy(
        delta_vector=np.array([1, 1, 1]), hamiltonian=egx_h, k_points=BZ_grid
    )
    free_energy_uniform_pairing = hamiltonians.free_energy_uniform_pairing(
        delta=1, hamiltonian=egx_h, k_points=BZ_grid
    )

    ndarrays_regression.check(
        {
            "free_energy": np.array(free_energy),
            "free_energy_uniform_pairing": np.array(free_energy_uniform_pairing),
        },
        default_tolerance=dict(atol=1e-8, rtol=1e-8),
    )
