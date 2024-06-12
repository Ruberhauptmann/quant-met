import os.path

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import (
    builds,
    floats,
    from_type,
    one_of,
    register_type_strategy,
)
from scipy import linalg

from quant_met import hamiltonians


@pytest.fixture()
def patch_abstract(monkeypatch):
    """Patch the abstract methods."""
    monkeypatch.setattr(hamiltonians.BaseHamiltonian, "__abstractmethods__", set())


register_type_strategy(
    hamiltonians.GrapheneHamiltonian,
    builds(
        hamiltonians.GrapheneHamiltonian,
        a=floats(min_value=0, exclude_min=True, allow_nan=False, allow_infinity=False),
        t_nn=floats(allow_nan=False, allow_infinity=False),
        mu=floats(allow_nan=False, allow_infinity=False),
        coulomb_gr=floats(allow_nan=False, allow_infinity=False),
    ),
)

register_type_strategy(
    hamiltonians.EGXHamiltonian,
    builds(
        hamiltonians.EGXHamiltonian,
        a=floats(min_value=0, exclude_min=True, allow_nan=False, allow_infinity=False),
        t_gr=floats(allow_nan=False, allow_infinity=False),
        t_x=floats(allow_nan=False, allow_infinity=False),
        mu=floats(allow_nan=False, allow_infinity=False),
        V=floats(allow_nan=False, allow_infinity=False),
        U_gr=floats(allow_nan=False, allow_infinity=False),
        U_x=floats(allow_nan=False, allow_infinity=False),
    ),
)


@given(
    sample=one_of(
        from_type(hamiltonians.GrapheneHamiltonian),
        from_type(hamiltonians.EGXHamiltonian),
    ),
    k=arrays(
        shape=(2,),
        dtype=float,
        elements=floats(allow_nan=False, allow_infinity=False),
    ),
)
def test_hamiltonians(sample: hamiltonians.BaseHamiltonian, k: npt.NDArray):
    h_k_space = sample.hamiltonian_k_space(k)[0]
    assert len(sample.coloumb_orbital_basis) == sample.number_of_bands
    assert h_k_space.shape[0] == sample.number_of_bands
    assert h_k_space.shape[1] == sample.number_of_bands
    assert linalg.ishermitian(h_k_space)


def test_save_graphene(tmp_path):
    graphene_h = hamiltonians.GrapheneHamiltonian(
        t_nn=1, a=np.sqrt(3), mu=-1, coulomb_gr=1
    )
    graphene_h.delta_orbital_basis = np.ones(graphene_h.number_of_bands)
    file_path = os.path.join(tmp_path, "test.hdf5")
    graphene_h.save(filename=file_path)
    sample_read = type(graphene_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        assert np.allclose(value, graphene_h.__dict__[key])


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
        print(
            base_hamiltonian._hamiltonian_k_space_one_point(
                k_point=np.array([0, 0]), matrix_in=np.array([[0, 0], [0, 0]])
            )
        )
