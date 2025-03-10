# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Test saving of Hamiltonians."""

from pathlib import Path

import numpy as np

from quant_met import mean_field, parameters


def test_save_graphene(tmp_path: Path) -> None:
    """Test whether a saved Graphene Hamiltonian is restored correctly."""
    graphene_h = mean_field.hamiltonians.Graphene(
        parameters=parameters.GrapheneParameters(
            hopping=1,
            lattice_constant=np.sqrt(3),
            chemical_potential=-1,
            hubbard_int_orbital_basis=[1.0, 1.0],
            delta=np.ones(2, dtype=np.complex128),
        )
    )
    file_path = tmp_path / "test.hdf5"
    graphene_h.save(filename=file_path)
    sample_read = type(graphene_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        if key not in ["name", "lattice"]:
            assert np.allclose(value, graphene_h.__dict__[key])


def test_save_graphene_with_beta_and_q(tmp_path: Path) -> None:
    """Test whether a saved Graphene Hamiltonian is restored correctly."""
    graphene_h = mean_field.hamiltonians.Graphene(
        parameters=parameters.GrapheneParameters(
            hopping=1,
            lattice_constant=np.sqrt(3),
            chemical_potential=-1,
            hubbard_int_orbital_basis=[1.0, 1.0],
            q=np.ones(2, dtype=np.float64),
            beta=100,
            delta=np.ones(2, dtype=np.complex128),
        )
    )
    file_path = tmp_path / "test.hdf5"
    graphene_h.save(filename=file_path)
    sample_read = type(graphene_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        if key not in ["name", "lattice"]:
            assert np.allclose(value, graphene_h.__dict__[key])


def test_save_dressed_graphene(tmp_path: Path) -> None:
    """Test whether a saved dressed Graphene Hamiltonian is restored correctly."""
    dressed_graphene_h = mean_field.hamiltonians.DressedGraphene(
        parameters=parameters.DressedGrapheneParameters(
            hopping_gr=1,
            hopping_x=0.01,
            hopping_x_gr_a=1,
            lattice_constant=np.sqrt(3),
            chemical_potential=0,
            hubbard_int_orbital_basis=[1.0, 1.0, 1.0],
            delta=np.ones(3, dtype=np.complex128),
        )
    )
    file_path = tmp_path / "test.hdf5"
    dressed_graphene_h.save(filename=file_path)
    sample_read = type(dressed_graphene_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        if key not in ["name", "lattice"]:
            assert np.allclose(value, dressed_graphene_h.__dict__[key])


def test_save_dressed_graphene_with_q_and_beta(tmp_path: Path) -> None:
    """Test whether a saved dressed Graphene Hamiltonian is restored correctly."""
    dressed_graphene_h = mean_field.hamiltonians.DressedGraphene(
        parameters=parameters.DressedGrapheneParameters(
            hopping_gr=1,
            hopping_x=0.01,
            hopping_x_gr_a=1,
            lattice_constant=np.sqrt(3),
            chemical_potential=0,
            hubbard_int_orbital_basis=[1.0, 1.0, 1.0],
            q=np.ones(2, dtype=np.float64),
            beta=100,
            delta=np.ones(3, dtype=np.complex128),
        )
    )
    file_path = tmp_path / "test.hdf5"
    dressed_graphene_h.save(filename=file_path)
    sample_read = type(dressed_graphene_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        if key not in ["name", "lattice"]:
            assert np.allclose(value, dressed_graphene_h.__dict__[key])


def test_save_one_band(tmp_path: Path) -> None:
    """Test whether a saved one band Hamiltonian is restored correctly."""
    one_band_h = mean_field.hamiltonians.OneBand(
        parameters=parameters.OneBandParameters(
            hopping=1,
            lattice_constant=1,
            chemical_potential=0,
            hubbard_int_orbital_basis=[1.0],
            delta=np.ones(1, dtype=np.complex128),
        )
    )
    file_path = tmp_path / "test.hdf5"
    one_band_h.save(filename=file_path)
    sample_read = type(one_band_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        if key not in ["name", "lattice"]:
            assert np.allclose(value, one_band_h.__dict__[key])


def test_save_one_band_with_q_and_beta(tmp_path: Path) -> None:
    """Test whether a saved one band Hamiltonian is restored correctly."""
    one_band_h = mean_field.hamiltonians.OneBand(
        parameters=parameters.OneBandParameters(
            hopping=1,
            lattice_constant=1,
            chemical_potential=0,
            hubbard_int_orbital_basis=[1.0],
            q=np.ones(2, dtype=np.float64),
            beta=100,
            delta=np.ones(1, dtype=np.complex128),
        )
    )
    file_path = tmp_path / "test.hdf5"
    one_band_h.save(filename=file_path)
    sample_read = type(one_band_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        if key not in ["name", "lattice"]:
            assert np.allclose(value, one_band_h.__dict__[key])


def test_save_two_band(tmp_path: Path) -> None:
    """Test whether via saved two band Hamiltonian is restored correctly."""
    one_band_h = mean_field.hamiltonians.TwoBand(
        parameters=parameters.TwoBandParameters(
            hopping=1,
            lattice_constant=1,
            chemical_potential=0,
            hubbard_int_orbital_basis=[1.0, 1.0],
            delta=np.ones(2, dtype=np.complex128),
        )
    )
    file_path = tmp_path / "test.hdf5"
    one_band_h.save(filename=file_path)
    sample_read = type(one_band_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        if key not in ["name", "lattice"]:
            assert np.allclose(value, one_band_h.__dict__[key])


def test_save_two_band_with_q_and_beta(tmp_path: Path) -> None:
    """Test whether via saved two band Hamiltonian is restored correctly."""
    one_band_h = mean_field.hamiltonians.TwoBand(
        parameters=parameters.TwoBandParameters(
            hopping=1,
            lattice_constant=1,
            chemical_potential=0,
            hubbard_int_orbital_basis=[1.0, 1.0],
            q=np.ones(2, dtype=np.float64),
            beta=100,
        )
    )
    file_path = tmp_path / "test.hdf5"
    one_band_h.save(filename=file_path)
    sample_read = type(one_band_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        if key not in ["name", "lattice"]:
            assert np.allclose(value, one_band_h.__dict__[key])


def test_save_three_band(tmp_path: Path) -> None:
    """Test whether a saved three band Hamiltonian is restored correctly."""
    one_band_h = mean_field.hamiltonians.ThreeBand(
        parameters=parameters.ThreeBandParameters(
            hopping=1,
            lattice_constant=1,
            chemical_potential=0,
            hubbard_int_orbital_basis=[1.0, 1.0, 1.0],
            delta=np.ones(3, dtype=np.complex128),
        )
    )
    file_path = tmp_path / "test.hdf5"
    one_band_h.save(filename=file_path)
    sample_read = type(one_band_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        if key not in ["name", "lattice"]:
            assert np.allclose(value, one_band_h.__dict__[key])


def test_save_three_band_with_q_and_beta(tmp_path: Path) -> None:
    """Test whether a saved three band Hamiltonian is restored correctly."""
    one_band_h = mean_field.hamiltonians.ThreeBand(
        parameters=parameters.ThreeBandParameters(
            hopping=1,
            lattice_constant=1,
            chemical_potential=0,
            hubbard_int_orbital_basis=[1.0, 1.0, 1.0],
            q=np.ones(2, dtype=np.float64),
            beta=100,
        )
    )
    file_path = tmp_path / "test.hdf5"
    one_band_h.save(filename=file_path)
    sample_read = type(one_band_h).from_file(filename=file_path)
    for key, value in vars(sample_read).items():
        if key not in ["name", "lattice"]:
            assert np.allclose(value, one_band_h.__dict__[key])
