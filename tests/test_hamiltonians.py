# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

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
from quant_met import mean_field, utils
from scipy import linalg


@pytest.fixture()
def patch_abstract(monkeypatch):
    """Patch the abstract methods."""
    monkeypatch.setattr(mean_field.BaseHamiltonian, "__abstractmethods__", set())


register_type_strategy(
    mean_field.GrapheneHamiltonian,
    builds(
        mean_field.GrapheneHamiltonian,
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
    mean_field.EGXHamiltonian,
    builds(
        mean_field.EGXHamiltonian,
        lattice_constant=floats(
            min_value=0,
            max_value=1e5,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        hopping_gr=floats(
            min_value=0,
            max_value=1e6,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        hopping_x=floats(
            min_value=0,
            max_value=1e6,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        mu=floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        hopping_x_gr_a=floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        coloumb_gr=floats(max_value=1e6, allow_nan=False, allow_infinity=False),
        coloumb_x=floats(max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
)


@given(
    sample=one_of(
        from_type(mean_field.GrapheneHamiltonian),
        from_type(mean_field.EGXHamiltonian),
    ),
    k=arrays(
        shape=tuples(integers(min_value=0, max_value=100), just(2)),
        dtype=float,
        elements=floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
)
def test_hamiltonians(sample: mean_field.BaseHamiltonian, k: npt.NDArray):
    sample.delta_orbital_basis = np.array([0 for _ in range(sample.number_of_bands)])

    bdg_energies = sample.diagonalize_bdg(k=k)[0].flatten()

    nonint_energies = np.array(
        [[+E, -E] for E in sample.diagonalize_nonint(k=k)[0].flatten()]
    ).flatten()

    h_k_space = sample.hamiltonian(k)
    if h_k_space.ndim == 2:
        h_k_space = np.expand_dims(h_k_space, axis=0)

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
    K = 4 * np.pi / (3 * lattice_constant) * np.array([1, 0])

    h_at_high_symmetry_points = [
        (Gamma, np.array([[-mu, -3 * t_gr], [-3 * t_gr, -mu]], dtype=np.complex64)),
        (K, np.array([[-mu, 0], [0, -mu]], dtype=np.complex64)),
    ]

    for k_point, h_compare in h_at_high_symmetry_points:
        graphene_h = mean_field.GrapheneHamiltonian(
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
        egx_h = mean_field.EGXHamiltonian(
            hopping_gr=t_gr,
            hopping_x=t_x,
            hopping_x_gr_a=V,
            lattice_constant=lattice_constant,
            mu=mu,
            coloumb_gr=0,
            coloumb_x=0,
        )
        h_generated = egx_h.hamiltonian(k_point)
        print(h_generated)
        print(h_compare)
        assert np.allclose(h_generated, h_compare)


def test_hamiltonian_derivative_graphene(ndarrays_regression):
    t_nn = 1
    mu = 0
    lattice_constant = np.sqrt(3)
    graphene_h = mean_field.GrapheneHamiltonian(
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
        * np.array([(np.sin(i * np.pi / 6), np.cos(i * np.pi / 6)) for i in [1, 3, 5, 7, 9, 11]])
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


def test_bdg_hamiltonian_derivative_graphene(ndarrays_regression):
    t_nn = 1
    mu = 0
    lattice_constant = np.sqrt(3)
    graphene_h = mean_field.GrapheneHamiltonian(
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
            * np.array([(np.sin(i * np.pi / 6), np.cos(i * np.pi / 6)) for i in [1, 3, 5, 7, 9, 11]])
    )
    BZ_grid = utils.generate_uniform_grid(
        10, 10, all_K_points[1], all_K_points[5], origin=np.array([0, 0])
    )

    h_der_x = graphene_h.bdg_hamiltonian_derivative(k=BZ_grid, direction="x")
    h_der_y = graphene_h.bdg_hamiltonian_derivative(k=BZ_grid, direction="y")
    h_der_x_one_point = graphene_h.bdg_hamiltonian_derivative(k=np.array([1, 1]), direction="x")
    h_der_y_one_point = graphene_h.bdg_hamiltonian_derivative(k=np.array([1, 1]), direction="y")

    ndarrays_regression.check(
        {
            "h_der_x": h_der_x,
            "h_der_y": h_der_y,
            "h_der_x_one_point": h_der_x_one_point,
            "h_der_y_one_point": h_der_y_one_point,
        },
        default_tolerance=dict(atol=1e-8, rtol=1e-8),
    )


def test_save_graphene(tmp_path):
    graphene_h = mean_field.GrapheneHamiltonian(t_nn=1, a=np.sqrt(3), mu=-1, coulomb_gr=1)
    graphene_h.delta_orbital_basis = np.ones(graphene_h.number_of_bands)
    file_path = Path(os.path.join(tmp_path, "test.hdf5"))
    graphene_h.save(filename=file_path)
    sample_read = type(graphene_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        assert np.allclose(value, graphene_h.__dict__[key])


def test_save_egx(tmp_path):
    egx_h = mean_field.EGXHamiltonian(
        hopping_gr=1,
        hopping_x=0.01,
        hopping_x_gr_a=1,
        lattice_constant=np.sqrt(3),
        mu=0,
        coloumb_gr=1,
        coloumb_x=1,
    )
    egx_h.delta_orbital_basis = np.ones(egx_h.number_of_bands)
    file_path = Path(os.path.join(tmp_path, "test.hdf5"))
    egx_h.save(filename=file_path)
    sample_read = type(egx_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        assert np.allclose(value, egx_h.__dict__[key])


def test_invalid_values():
    with pytest.raises(ValueError):
        print(mean_field.GrapheneHamiltonian(t_nn=1, a=-1, mu=1, coulomb_gr=1))
    with pytest.raises(ValueError):
        print(mean_field.GrapheneHamiltonian(t_nn=np.nan, a=1, mu=1, coulomb_gr=1))
    with pytest.raises(ValueError):
        print(mean_field.GrapheneHamiltonian(t_nn=1, a=np.nan, mu=1, coulomb_gr=1))
    with pytest.raises(ValueError):
        print(mean_field.GrapheneHamiltonian(t_nn=1, a=1, mu=np.nan, coulomb_gr=1))
    with pytest.raises(ValueError):
        print(mean_field.GrapheneHamiltonian(t_nn=1, a=1, mu=1, coulomb_gr=np.nan))
    with pytest.raises(ValueError):
        print(mean_field.GrapheneHamiltonian(t_nn=np.inf, a=1, mu=1, coulomb_gr=1))
    with pytest.raises(ValueError):
        print(mean_field.GrapheneHamiltonian(t_nn=1, a=np.inf, mu=1, coulomb_gr=1))
    with pytest.raises(ValueError):
        print(mean_field.GrapheneHamiltonian(t_nn=1, a=1, mu=np.inf, coulomb_gr=1))
    with pytest.raises(ValueError):
        print(mean_field.GrapheneHamiltonian(t_nn=1, a=1, mu=1, coulomb_gr=np.nan))
    with pytest.raises(ValueError):
        h = mean_field.GrapheneHamiltonian(t_nn=1, a=1, mu=1, coulomb_gr=1)
        h.hamiltonian(k=np.array([np.nan, np.nan]))
    with pytest.raises(ValueError):
        h = mean_field.GrapheneHamiltonian(t_nn=1, a=1, mu=1, coulomb_gr=1)
        h.hamiltonian(k=np.array([[np.nan, np.inf]]))
    with pytest.raises(ValueError):
        h = mean_field.GrapheneHamiltonian(t_nn=1, a=1, mu=1, coulomb_gr=1)
        h.bdg_hamiltonian(k=np.array([np.nan, np.nan]))
    with pytest.raises(ValueError):
        h = mean_field.GrapheneHamiltonian(t_nn=1, a=1, mu=1, coulomb_gr=1)
        h.hamiltonian_derivative(k=np.array([np.nan, np.nan]), direction="x")


def test_base_hamiltonian(patch_abstract) -> None:
    """Test whether the Hamiltonian Base class fullfills relevant promises.

    Args:
    ----
        patch_abstract: Fixture to be able to initialise BaseHamiltonian.

    Returns
    -------
        None

    """
    base_hamiltonian = mean_field.BaseHamiltonian()
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.number_of_bands)
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.coloumb_orbital_basis)
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.delta_orbital_basis)
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.hamiltonian(np.array([1, 2, 3])))
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.hamiltonian_derivative(np.array([1, 2, 3]), direction='x'))
    with pytest.raises(NotImplementedError):
        base_hamiltonian.delta_orbital_basis = np.array([0])
