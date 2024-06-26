import os.path
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import (
    builds,
    floats,
    from_type,
    integers,
    just,
    one_of,
    register_type_strategy,
    tuples,
)
from scipy import linalg

from quant_met import hamiltonians, utils


@pytest.fixture()
def patch_abstract(monkeypatch):
    """Patch the abstract methods."""
    monkeypatch.setattr(hamiltonians.BaseHamiltonian, "__abstractmethods__", set())


register_type_strategy(
    hamiltonians.GrapheneHamiltonian,
    builds(
        hamiltonians.GrapheneHamiltonian,
        a=floats(
            min_value=0,
            max_value=1e4,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        t_nn=floats(
            min_value=0,
            max_value=1e6,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        mu=floats(max_value=1e6, allow_nan=False, allow_infinity=False),
        coulomb_gr=floats(max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
)

register_type_strategy(
    hamiltonians.EGXHamiltonian,
    builds(
        hamiltonians.EGXHamiltonian,
        a=floats(
            min_value=0,
            max_value=1e5,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        t_gr=floats(
            min_value=0,
            max_value=1e6,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        t_x=floats(
            min_value=0,
            max_value=1e6,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        mu=floats(max_value=1e6, allow_nan=False, allow_infinity=False),
        V=floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        U_gr=floats(max_value=1e6, allow_nan=False, allow_infinity=False),
        U_x=floats(max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
)


@given(
    sample=one_of(
        from_type(hamiltonians.GrapheneHamiltonian),
        from_type(hamiltonians.EGXHamiltonian),
    ),
    k=arrays(
        shape=tuples(integers(min_value=0, max_value=int(100)), just(2)),
        dtype=float,
        elements=floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    ),
)
def test_hamiltonians(sample: hamiltonians.BaseHamiltonian, k: npt.NDArray):
    sample.delta_orbital_basis = np.array([0 for _ in range(sample.number_of_bands)])

    bdg_energies = sample.diagonalize_bdg(k=k)[0].flatten()

    nonint_energies = np.array(
        [[+E, -E] for E in sample.diagonalize_nonint(k=k)[0].flatten()]
    ).flatten()

    h_k_space = sample.hamiltonian(k)

    assert np.allclose(
        np.sort(np.nan_to_num(bdg_energies.flatten())),
        np.sort(np.nan_to_num(nonint_energies)),
    )
    assert len(sample.coloumb_orbital_basis) == sample.number_of_bands
    for h in h_k_space:
        assert h.shape[0] == sample.number_of_bands
        assert h.shape[1] == sample.number_of_bands
        assert linalg.ishermitian(h)


def test_hamiltonian_k_space_graphene():
    t_gr = 1
    mu = 1
    lattice_constant = np.sqrt(3)
    Gamma = np.array([0, 0])
    M = np.pi / lattice_constant * np.array([1, 1 / np.sqrt(3)])
    K = 4 * np.pi / (3 * lattice_constant) * np.array([1, 0])

    h_at_high_symmetry_points = [
        (Gamma, np.array([[-mu, -3 * t_gr], [-3 * t_gr, -mu]], dtype=np.complex64)),
        (K, np.array([[-mu, 0], [0, -mu]], dtype=np.complex64)),
    ]

    for k_point, h_compare in h_at_high_symmetry_points:
        graphene_h = hamiltonians.GrapheneHamiltonian(
            t_nn=t_gr, a=lattice_constant, mu=mu, coulomb_gr=0
        )
        h_generated = graphene_h.hamiltonian(k_point)
        assert np.allclose(h_generated, h_compare)


def test_hamiltonian_k_space_egx():
    t_gr = 1
    t_x = 0.01
    V = 1
    mu = 1
    lattice_constant = np.sqrt(3)
    Gamma = np.array([0, 0])
    M = np.pi / lattice_constant * np.array([1, 1 / np.sqrt(3)])
    K = 4 * np.pi / (3 * lattice_constant) * np.array([1, 0])

    h_at_high_symmetry_points = [
        (
            Gamma,
            np.array(
                [[-mu, -3 * t_gr, V], [-3 * t_gr, -mu, 0], [V, 0, -mu - 6 * t_x]],
                dtype=np.complex64,
            ),
        ),
    ]

    for k_point, h_compare in h_at_high_symmetry_points:
        egx_h = hamiltonians.EGXHamiltonian(
            t_gr=t_gr, t_x=t_x, V=V, a=lattice_constant, mu=mu, U_gr=0, U_x=0
        )
        h_generated = egx_h.hamiltonian(k_point)
        assert np.allclose(h_generated, h_compare)


def test_hamiltonian_derivative_graphene(ndarrays_regression):
    t_nn = 1
    mu = 0
    lattice_constant = np.sqrt(3)
    graphene_h = hamiltonians.GrapheneHamiltonian(
        t_nn=t_nn,
        a=lattice_constant,
        mu=mu,
        coulomb_gr=1,
        delta=np.array([1, 1]),
    )
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
    BZ_grid = utils.generate_uniform_grid(
        10, 10, all_K_points[1], all_K_points[5], origin=np.array([0, 0])
    )

    h_der_x = graphene_h.hamiltonian_derivative(k=BZ_grid, direction="x")
    h_der_y = graphene_h.hamiltonian_derivative(k=BZ_grid, direction="y")

    ndarrays_regression.check(
        {
            "h_der_x": h_der_x,
            "h_der_y": h_der_y,
        },
        default_tolerance=dict(atol=1e-8, rtol=1e-8),
    )


def test_save_graphene(tmp_path):
    graphene_h = hamiltonians.GrapheneHamiltonian(
        t_nn=1, a=np.sqrt(3), mu=-1, coulomb_gr=1
    )
    graphene_h.delta_orbital_basis = np.ones(graphene_h.number_of_bands)
    file_path = Path(os.path.join(tmp_path, "test.hdf5"))
    graphene_h.save(filename=file_path)
    sample_read = type(graphene_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        assert np.allclose(value, graphene_h.__dict__[key])


def test_save_egx(tmp_path):
    egx_h = hamiltonians.EGXHamiltonian(
        t_gr=1, t_x=0.01, V=1, a=np.sqrt(3), mu=-1, U_gr=1, U_x=1
    )
    egx_h.delta_orbital_basis = np.ones(egx_h.number_of_bands)
    file_path = Path(os.path.join(tmp_path, "test.hdf5"))
    egx_h.save(filename=file_path)
    sample_read = type(egx_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        assert np.allclose(value, egx_h.__dict__[key])


def test_invalid_values():
    with pytest.raises(ValueError):
        print(hamiltonians.GrapheneHamiltonian(t_nn=1, a=-1, mu=1, coulomb_gr=1))
    with pytest.raises(ValueError):
        print(hamiltonians.GrapheneHamiltonian(t_nn=np.nan, a=1, mu=1, coulomb_gr=1))
    with pytest.raises(ValueError):
        print(hamiltonians.GrapheneHamiltonian(t_nn=1, a=np.nan, mu=1, coulomb_gr=1))
    with pytest.raises(ValueError):
        print(hamiltonians.GrapheneHamiltonian(t_nn=1, a=1, mu=np.nan, coulomb_gr=1))
    with pytest.raises(ValueError):
        print(hamiltonians.GrapheneHamiltonian(t_nn=1, a=1, mu=1, coulomb_gr=np.nan))
    with pytest.raises(ValueError):
        print(hamiltonians.GrapheneHamiltonian(t_nn=np.inf, a=1, mu=1, coulomb_gr=1))
    with pytest.raises(ValueError):
        print(hamiltonians.GrapheneHamiltonian(t_nn=1, a=np.inf, mu=1, coulomb_gr=1))
    with pytest.raises(ValueError):
        print(hamiltonians.GrapheneHamiltonian(t_nn=1, a=1, mu=np.inf, coulomb_gr=1))
    with pytest.raises(ValueError):
        print(hamiltonians.GrapheneHamiltonian(t_nn=1, a=1, mu=1, coulomb_gr=np.nan))
    with pytest.raises(ValueError):
        h = hamiltonians.GrapheneHamiltonian(t_nn=1, a=1, mu=1, coulomb_gr=1)
        h.hamiltonian(k=np.array([np.nan, np.nan]))
    with pytest.raises(ValueError):
        h = hamiltonians.GrapheneHamiltonian(t_nn=1, a=1, mu=1, coulomb_gr=1)
        h.hamiltonian(k=np.array([[np.nan, np.inf]]))
    with pytest.raises(ValueError):
        h = hamiltonians.GrapheneHamiltonian(t_nn=1, a=1, mu=1, coulomb_gr=1)
        h.bdg_hamiltonian(k=np.array([np.nan, np.nan]))
    with pytest.raises(ValueError):
        h = hamiltonians.GrapheneHamiltonian(t_nn=1, a=1, mu=1, coulomb_gr=1)
        h.hamiltonian_derivative(k=np.array([np.nan, np.nan]), direction="x")


def test_base_hamiltonian(patch_abstract) -> None:
    """Test whether the Hamiltonian Base class fullfills relevant promises.

    Args:
        patch_abstract: Fixture to be able to initialise BaseHamiltonian.

    Returns:
        None
    """
    base_hamiltonian = hamiltonians.BaseHamiltonian()
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.number_of_bands)
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.coloumb_orbital_basis)
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.delta_orbital_basis)
    with pytest.raises(NotImplementedError):
        base_hamiltonian.delta_orbital_basis = np.array([0])
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian._hamiltonian_one_point(k_point=np.array([0, 0])))
    with pytest.raises(NotImplementedError):
        print(
            base_hamiltonian._hamiltonian_derivative_one_point(
                k_point=np.array([0, 0]), directions="x"
            )
        )
