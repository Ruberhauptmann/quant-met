# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Tests for the geometry module."""

import numpy as np
import pytest

from quant_met import geometry


@pytest.fixture
def _patch_abstract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the abstract methods."""
    monkeypatch.setattr(geometry.BaseLattice, "__abstractmethods__", set())


def test_generate_bz_path() -> None:
    """Test generating of BZ paths."""
    graphene_lattice = geometry.GrapheneLattice(lattice_constant=np.sqrt(3))
    band_path, band_path_plot, ticks, labels = graphene_lattice.generate_high_symmetry_path(
        number_of_points=1000
    )

    assert labels == ["$M$", "$\\Gamma$", "$K$", "$M$"]
    assert ticks[0] == 0.0
    assert band_path_plot[0] == 0.0


@pytest.mark.usefixtures("_patch_abstract")
def test_base_lattice() -> None:
    """Test BaseLattice functions."""
    base_lattice = geometry.BaseLattice()
    with pytest.raises(NotImplementedError):
        print(base_lattice.lattice_constant)
    with pytest.raises(NotImplementedError):
        print(base_lattice.bz_corners)
    with pytest.raises(NotImplementedError):
        print(base_lattice.high_symmetry_points)
