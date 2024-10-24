# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Test Hamiltonian classes."""

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
    none,
    one_of,
    register_type_strategy,
    tuples,
)
from pydantic import BaseModel
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture
from quant_met import geometry, mean_field, parameters
from quant_met.mean_field.hamiltonians import BaseHamiltonian
from scipy import linalg


@pytest.fixture()
def _patch_abstract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the abstract methods."""
    monkeypatch.setattr(mean_field.hamiltonians.BaseHamiltonian, "__abstractmethods__", set())


def _hamiltonian_factory(classname: str, input_parameters: BaseModel) -> BaseHamiltonian:
    """Create a hamiltonian by its class name."""
    from quant_met.mean_field import hamiltonians

    cls = getattr(hamiltonians, classname)
    h: BaseHamiltonian = cls(input_parameters)
    return h


register_type_strategy(
    parameters.OneBandParameters,
    builds(
        parameters.OneBandParameters,
        lattice_constant=floats(
            min_value=0,
            max_value=1e4,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        hopping=floats(
            min_value=0,
            max_value=1e6,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        chemical_potential=floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        hubbard_int=floats(max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
)

register_type_strategy(
    parameters.GrapheneParameters,
    builds(
        parameters.GrapheneParameters,
        lattice_constant=floats(
            min_value=0,
            max_value=1e4,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        hopping=floats(
            min_value=0,
            max_value=1e6,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
        ),
        chemical_potential=floats(max_value=1e6, allow_nan=False, allow_infinity=False),
        hubbard_int=floats(max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
)

register_type_strategy(
    parameters.DressedGrapheneParameters,
    builds(
        parameters.DressedGrapheneParameters,
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
        chemical_potential=floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        hopping_x_gr_a=floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        hubbard_int_gr=floats(max_value=1e6, allow_nan=False, allow_infinity=False),
        hubbard_int_x=floats(max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
)


@given(
    sample_parameters=one_of(
        from_type(parameters.DressedGrapheneParameters),
        from_type(parameters.GrapheneParameters),
        from_type(parameters.OneBandParameters),
    ),
    k=arrays(
        shape=tuples(integers(min_value=0, max_value=100), just(2)),
        dtype=float,
        elements=floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
    q=one_of(
        arrays(
            shape=tuples(just(2)),
            dtype=float,
            elements=floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        ),
        none(),
    ),
)
def test_hamiltonians(sample_parameters: BaseModel, k: npt.NDArray, q: npt.NDArray | None) -> None:
    """Test Hamiltonians with random parameters."""
    sample = _hamiltonian_factory(
        input_parameters=sample_parameters, classname=sample_parameters.name
    )

    assert sample.name == sample_parameters.name

    sample.delta_orbital_basis = np.array([0 for _ in range(sample.number_of_bands)])

    bdg_energies = sample.diagonalize_bdg(k=k)[0].flatten()

    if q is None:
        nonint_energies = np.array(
            [[+E, -E] for E in sample.diagonalize_nonint(k=k)[0].flatten()]
        ).flatten()
        assert np.allclose(
            np.sort(np.nan_to_num(bdg_energies.flatten())),
            np.sort(np.nan_to_num(nonint_energies)),
        )

    h_k_space = sample.hamiltonian(k)
    if h_k_space.ndim == 2:
        h_k_space = np.expand_dims(h_k_space, axis=0)

    assert len(sample.hubbard_int_orbital_basis) == sample.number_of_bands
    for h in h_k_space:
        assert h.shape[0] == sample.number_of_bands
        assert h.shape[1] == sample.number_of_bands
        assert linalg.ishermitian(h)

    assert sample.beta == 1000.0


def test_hamiltonian_k_space_graphene() -> None:
    """Test Graphene Hamiltonians at some k points."""
    t_gr = 1
    chemical_potential = 1
    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))
    h_at_high_symmetry_points = [
        (
            graphene_lattice.Gamma,
            np.array(
                [[-chemical_potential, -3 * t_gr], [-3 * t_gr, -chemical_potential]],
                dtype=np.complex64,
            ),
        ),
        (
            graphene_lattice.K,
            np.array([[-chemical_potential, 0], [0, -chemical_potential]], dtype=np.complex64),
        ),
    ]

    for k_point, h_compare in h_at_high_symmetry_points:
        graphene_h = mean_field.hamiltonians.Graphene(
            parameters=parameters.GrapheneParameters(
                hopping=t_gr,
                lattice_constant=graphene_lattice.lattice_constant,
                chemical_potential=chemical_potential,
                hubbard_int=0,
            )
        )
        h_generated = graphene_h.hamiltonian(k_point)
        assert np.allclose(h_generated, h_compare)


def test_hamiltonian_k_space_egx() -> None:
    """Test dressed Graphene Hamiltonian at some k points."""
    t_gr = 1
    t_x = 0.01
    v = 1
    chemical_potential = 1

    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))
    h_at_high_symmetry_points = [
        (
            graphene_lattice.Gamma,
            np.array(
                [
                    [-chemical_potential, -3 * t_gr, v],
                    [-3 * t_gr, -chemical_potential, 0],
                    [v, 0, -chemical_potential - 6 * t_x],
                ],
                dtype=np.complex64,
            ),
        ),
    ]

    for k_point, h_compare in h_at_high_symmetry_points:
        egx_h = mean_field.hamiltonians.DressedGraphene(
            parameters=parameters.DressedGrapheneParameters(
                hopping_gr=t_gr,
                hopping_x=t_x,
                hopping_x_gr_a=v,
                lattice_constant=graphene_lattice.lattice_constant,
                chemical_potential=chemical_potential,
                hubbard_int_gr=0,
                hubbard_int_x=0,
            )
        )
        h_generated = egx_h.hamiltonian(k_point)
        assert np.allclose(h_generated, h_compare)


def test_hamiltonian_k_space_one_band() -> None:
    """Test one band Hamiltonian at some k points."""
    chemical_potential = 1

    square_lattice = geometry.SquareLattice(lattice_constant=1.0)
    h_at_high_symmetry_points = [
        (
            square_lattice.Gamma,
            np.array(
                [[-4 - chemical_potential]],
                dtype=np.complex64,
            ),
        ),
    ]

    for k_point, h_compare in h_at_high_symmetry_points:
        one_band_h = mean_field.hamiltonians.OneBand(
            parameters=parameters.OneBandParameters(
                hopping=1,
                lattice_constant=square_lattice.lattice_constant,
                chemical_potential=chemical_potential,
                hubbard_int=0,
            )
        )
        h_generated = one_band_h.hamiltonian(k_point)
        assert np.allclose(h_generated, h_compare)


def test_gap_equation_egx_nonint() -> None:
    """Test gap equation for dressed Graphene model."""
    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))

    egx_h = mean_field.hamiltonians.DressedGraphene(
        parameters=parameters.DressedGrapheneParameters(
            hopping_gr=1,
            hopping_x=0.01,
            hopping_x_gr_a=1,
            lattice_constant=graphene_lattice.lattice_constant,
            chemical_potential=0,
            hubbard_int_gr=0,
            hubbard_int_x=0,
        )
    )
    assert np.allclose(
        egx_h.gap_equation(k=graphene_lattice.generate_bz_grid(ncols=30, nrows=30)),
        np.zeros(3, dtype=np.complex64),
    )


def test_density_of_states(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for density of states."""
    hopping = 1
    chemical_potential = 0

    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))
    bz_grid = graphene_lattice.generate_bz_grid(10, 10)
    graphene_h = mean_field.hamiltonians.Graphene(
        parameters=parameters.GrapheneParameters(
            hopping=hopping,
            lattice_constant=graphene_lattice.lattice_constant,
            chemical_potential=chemical_potential,
            hubbard_int=1,
            delta=np.array([1, 1], dtype=np.complex64),
        )
    )

    dos = graphene_h.calculate_density_of_states(k=bz_grid)

    ndarrays_regression.check(
        {
            "DOS": dos,
        },
        default_tolerance={"atol": 1e-8, "rtol": 1e-8},
    )


def test_spectral_gap(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for calculation of spectral gap."""
    hopping = 1
    chemical_potential = 0

    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))
    bz_grid = graphene_lattice.generate_bz_grid(10, 10)
    egx_h = mean_field.hamiltonians.DressedGraphene(
        parameters=parameters.DressedGrapheneParameters(
            hopping_gr=hopping,
            hopping_x=0.01,
            hopping_x_gr_a=1.0,
            lattice_constant=graphene_lattice.lattice_constant,
            chemical_potential=chemical_potential,
            hubbard_int_gr=1,
            hubbard_int_x=1,
            delta=np.array([0, 0, 0], dtype=np.complex64),
        )
    )
    spectral_gap_zero_gap = egx_h.calculate_spectral_gap(k=bz_grid)

    egx_h = mean_field.hamiltonians.DressedGraphene(
        parameters=parameters.DressedGrapheneParameters(
            hopping_gr=hopping,
            hopping_x=0.01,
            hopping_x_gr_a=1.0,
            lattice_constant=graphene_lattice.lattice_constant,
            chemical_potential=chemical_potential,
            hubbard_int_gr=1,
            hubbard_int_x=1,
            delta=np.array([1, 1, 1], dtype=np.complex64),
        )
    )
    spectral_gap_finite_gap = egx_h.calculate_spectral_gap(k=bz_grid)

    ndarrays_regression.check(
        {
            "zero gap": spectral_gap_zero_gap,
            "finite gap": spectral_gap_finite_gap,
        },
        default_tolerance={"atol": 1e-8, "rtol": 1e-8},
    )


def test_hamiltonian_derivative_graphene(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for the derivative of the Graphene Hamiltonian."""
    hopping = 1
    chemical_potential = 0

    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))
    bz_grid = graphene_lattice.generate_bz_grid(10, 10)
    graphene_h = mean_field.hamiltonians.Graphene(
        parameters=parameters.GrapheneParameters(
            hopping=hopping,
            lattice_constant=graphene_lattice.lattice_constant,
            chemical_potential=chemical_potential,
            hubbard_int=1,
            delta=np.array([1, 1], dtype=np.complex64),
        )
    )

    h_der_x = graphene_h.hamiltonian_derivative(k=bz_grid, direction="x")
    h_der_y = graphene_h.hamiltonian_derivative(k=bz_grid, direction="y")

    ndarrays_regression.check(
        {
            "h_der_x": h_der_x,
            "h_der_y": h_der_y,
        },
        default_tolerance={"atol": 1e-8, "rtol": 1e-8},
    )


def test_hamiltonian_derivative_one_band(ndarrays_regression: NDArraysRegressionFixture) -> None:
    """Regression test for the derivative of the one band Hamiltonian."""
    hopping = 1
    chemical_potential = 0

    square_lattice = geometry.SquareLattice(lattice_constant=1)
    bz_grid = square_lattice.generate_bz_grid(10, 10)
    one_band_h = mean_field.hamiltonians.OneBand(
        parameters=parameters.OneBandParameters(
            hopping=hopping,
            lattice_constant=square_lattice.lattice_constant,
            chemical_potential=chemical_potential,
            hubbard_int=1,
            delta=np.array([1], dtype=np.complex64),
        )
    )

    h_der_x = one_band_h.hamiltonian_derivative(k=bz_grid, direction="x")
    h_der_y = one_band_h.hamiltonian_derivative(k=bz_grid, direction="y")
    h_der_x_one_point = one_band_h.hamiltonian_derivative(k=np.array([1, 1]), direction="x")
    h_der_y_one_point = one_band_h.hamiltonian_derivative(k=np.array([1, 1]), direction="y")

    ndarrays_regression.check(
        {
            "h_der_x": h_der_x,
            "h_der_y": h_der_y,
            "h_der_x_one_point": h_der_x_one_point,
            "h_der_y_one_point": h_der_y_one_point,
        },
        default_tolerance={"atol": 1e-8, "rtol": 1e-8},
    )


def test_bdg_hamiltonian_derivative_graphene(
    ndarrays_regression: NDArraysRegressionFixture,
) -> None:
    """Test the derivative of the Graphene BdG Hamiltonian."""
    hopping = 1
    chemical_potential = 0

    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))
    bz_grid = graphene_lattice.generate_bz_grid(10, 10)
    graphene_h = mean_field.hamiltonians.Graphene(
        parameters=parameters.GrapheneParameters(
            hopping=hopping,
            lattice_constant=graphene_lattice.lattice_constant,
            chemical_potential=chemical_potential,
            hubbard_int=1,
            delta=np.array([1, 1], dtype=np.complex64),
        )
    )

    h_der_x = graphene_h.bdg_hamiltonian_derivative(k=bz_grid, direction="x")
    h_der_y = graphene_h.bdg_hamiltonian_derivative(k=bz_grid, direction="y")
    h_der_x_one_point = graphene_h.bdg_hamiltonian_derivative(k=np.array([1, 1]), direction="x")
    h_der_y_one_point = graphene_h.bdg_hamiltonian_derivative(k=np.array([1, 1]), direction="y")

    ndarrays_regression.check(
        {
            "h_der_x": h_der_x,
            "h_der_y": h_der_y,
            "h_der_x_one_point": h_der_x_one_point,
            "h_der_y_one_point": h_der_y_one_point,
        },
        default_tolerance={"atol": 1e-8, "rtol": 1e-8},
    )


def test_save_graphene(tmp_path: Path) -> None:
    """Test whether a saved Graphene Hamiltonian is restored correctly."""
    graphene_h = mean_field.hamiltonians.Graphene(
        parameters=parameters.GrapheneParameters(
            hopping=1,
            lattice_constant=np.sqrt(3),
            chemical_potential=-1,
            hubbard_int=1,
            delta=np.ones(2, dtype=np.complex64),
        )
    )
    file_path = tmp_path / "test.hdf5"
    graphene_h.save(filename=file_path)
    sample_read = type(graphene_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        if not key.startswith("_"):
            assert np.allclose(value, graphene_h.__dict__[key])


def test_save_graphene_with_beta_and_q(tmp_path: Path) -> None:
    """Test whether a saved Graphene Hamiltonian is restored correctly."""
    graphene_h = mean_field.hamiltonians.Graphene(
        parameters=parameters.GrapheneParameters(
            hopping=1,
            lattice_constant=np.sqrt(3),
            chemical_potential=-1,
            hubbard_int=1,
            q=np.ones(2, dtype=np.float64),
            beta=100,
            delta=np.ones(2, dtype=np.complex64),
        )
    )
    file_path = tmp_path / "test.hdf5"
    graphene_h.save(filename=file_path)
    sample_read = type(graphene_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        if not key.startswith("_"):
            assert np.allclose(value, graphene_h.__dict__[key])


def test_save_egx(tmp_path: Path) -> None:
    """Test whether a saved dressed Graphene Hamiltonian is restored correctly."""
    egx_h = mean_field.hamiltonians.DressedGraphene(
        parameters=parameters.DressedGrapheneParameters(
            hopping_gr=1,
            hopping_x=0.01,
            hopping_x_gr_a=1,
            lattice_constant=np.sqrt(3),
            chemical_potential=0,
            hubbard_int_gr=1,
            hubbard_int_x=1,
            delta=np.ones(3, dtype=np.complex64),
        )
    )
    file_path = tmp_path / "test.hdf5"
    egx_h.save(filename=file_path)
    sample_read = type(egx_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        if not key.startswith("_"):
            assert np.allclose(value, egx_h.__dict__[key])


def test_save_egx_with_q_and_beta(tmp_path: Path) -> None:
    """Test whether a saved dressed Graphene Hamiltonian is restored correctly."""
    egx_h = mean_field.hamiltonians.DressedGraphene(
        parameters=parameters.DressedGrapheneParameters(
            hopping_gr=1,
            hopping_x=0.01,
            hopping_x_gr_a=1,
            lattice_constant=np.sqrt(3),
            chemical_potential=0,
            hubbard_int_gr=1,
            hubbard_int_x=1,
            q=np.ones(2, dtype=np.float64),
            beta=100,
            delta=np.ones(3, dtype=np.complex64),
        )
    )
    file_path = tmp_path / "test.hdf5"
    egx_h.save(filename=file_path)
    sample_read = type(egx_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        if not key.startswith("_"):
            assert np.allclose(value, egx_h.__dict__[key])


def test_save_one_band(tmp_path: Path) -> None:
    """Test whether a saved one band Hamiltonian is restored correctly."""
    one_band_h = mean_field.hamiltonians.OneBand(
        parameters=parameters.OneBandParameters(
            hopping=1,
            lattice_constant=1,
            chemical_potential=0,
            hubbard_int=1,
            delta=np.ones(1, dtype=np.complex64),
        )
    )
    file_path = tmp_path / "test.hdf5"
    one_band_h.save(filename=file_path)
    sample_read = type(one_band_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        if not key.startswith("_"):
            assert np.allclose(value, one_band_h.__dict__[key])


def test_save_one_band_with_q_and_beta(tmp_path: Path) -> None:
    """Test whether a saved one band Hamiltonian is restored correctly."""
    one_band_h = mean_field.hamiltonians.OneBand(
        parameters=parameters.OneBandParameters(
            hopping=1,
            lattice_constant=1,
            chemical_potential=0,
            hubbard_int=1,
            q=np.ones(2, dtype=np.float64),
            beta=100,
            delta=np.ones(1, dtype=np.complex64),
        )
    )
    file_path = tmp_path / "test.hdf5"
    one_band_h.save(filename=file_path)
    sample_read = type(one_band_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        if not key.startswith("_"):
            assert np.allclose(value, one_band_h.__dict__[key])


def test_invalid_values() -> None:
    """Test that invalid values are correctly identified."""
    with pytest.raises(ValueError, match="Lattice constant must be positive"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=1, lattice_constant=-1, chemical_potential=1, hubbard_int=1
                )
            )
        )
    with pytest.raises(ValueError, match="Lattice constant must be positive"):
        print(
            mean_field.hamiltonians.DressedGraphene(
                parameters=parameters.DressedGrapheneParameters(
                    hopping_gr=1,
                    hopping_x=1,
                    hopping_x_gr_a=1,
                    lattice_constant=-1,
                    chemical_potential=1,
                    hubbard_int_gr=1,
                    hubbard_int_x=1,
                )
            )
        )
    with pytest.raises(ValueError, match="Lattice constant must be positive"):
        print(
            mean_field.hamiltonians.OneBand(
                parameters=parameters.OneBandParameters(
                    hopping=1, lattice_constant=-1, chemical_potential=1, hubbard_int=1
                )
            )
        )
    with pytest.raises(ValueError, match="Hopping must not be NaN"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=np.nan, lattice_constant=1, chemical_potential=1, hubbard_int=1
                )
            )
        )
    with pytest.raises(ValueError, match="Lattice constant must not be NaN"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=1, lattice_constant=np.nan, chemical_potential=1, hubbard_int=1
                )
            )
        )
    with pytest.raises(ValueError, match="Chemical potential must not be NaN"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=1, lattice_constant=1, chemical_potential=np.nan, hubbard_int=1
                )
            )
        )
    with pytest.raises(ValueError, match="Hubbard interaction must not be NaN"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=1, lattice_constant=1, chemical_potential=1, hubbard_int=np.nan
                )
            )
        )
    with pytest.raises(ValueError, match="Hopping must not be Infinity"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=np.inf, lattice_constant=1, chemical_potential=1, hubbard_int=1
                )
            )
        )
    with pytest.raises(ValueError, match="Lattice constant must not be Infinity"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=1, lattice_constant=np.inf, chemical_potential=1, hubbard_int=1
                )
            )
        )
    with pytest.raises(ValueError, match="Chemical potential must not be Infinity"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=1, lattice_constant=1, chemical_potential=np.inf, hubbard_int=1
                )
            )
        )
    with pytest.raises(ValueError, match="Hubbard interaction must not be NaN"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=1, lattice_constant=1, chemical_potential=1, hubbard_int=np.nan
                )
            )
        )
    with pytest.raises(ValueError, match="k is NaN or Infinity"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=1, lattice_constant=1, chemical_potential=1, hubbard_int=1
                )
            ).hamiltonian(k=np.array([np.nan, np.nan]))
        )
    with pytest.raises(ValueError, match="k is NaN or Infinity"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=1, lattice_constant=1, chemical_potential=1, hubbard_int=1
                )
            ).hamiltonian(k=np.array([[np.nan, np.inf]]))
        )
    with pytest.raises(ValueError, match="k is NaN or Infinity"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=1, lattice_constant=1, chemical_potential=1, hubbard_int=1
                )
            ).bdg_hamiltonian(k=np.array([np.nan, np.nan]))
        )
    with pytest.raises(ValueError, match="k is NaN or Infinity"):
        print(
            mean_field.hamiltonians.Graphene(
                parameters=parameters.GrapheneParameters(
                    hopping=1, lattice_constant=1, chemical_potential=1, hubbard_int=1
                )
            ).hamiltonian_derivative(k=np.array([np.nan, np.nan]), direction="x")
        )


@pytest.mark.usefixtures("_patch_abstract")
def test_base_hamiltonian() -> None:
    """Test that the methods in the BaseHamiltonian class raises correct errors."""
    base_hamiltonian = mean_field.hamiltonians.BaseHamiltonian()
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.name)
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.q)
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.beta)
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.lattice)
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.from_file(Path("test.hdf5")))
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.number_of_bands)
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.hubbard_int_orbital_basis)
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.delta_orbital_basis)
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.hamiltonian(np.array([1, 2, 3])))
    with pytest.raises(NotImplementedError):
        print(base_hamiltonian.hamiltonian_derivative(np.array([1, 2, 3]), direction="x"))
    with pytest.raises(NotImplementedError):
        base_hamiltonian.delta_orbital_basis = np.array([0])
