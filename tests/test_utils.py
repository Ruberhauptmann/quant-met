import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers

from quant_met import utils


@given(
    ncols=integers(min_value=2, max_value=1000),
    nrows=integers(min_value=2, max_value=1000),
    corner_1=arrays(shape=2, dtype=np.float64),
    corner_2=arrays(shape=2, dtype=np.float64),
    origin=arrays(shape=2, dtype=np.float64),
)
def test_generate_uniform_grid_samples(ncols, nrows, corner_1, corner_2, origin):
    assume(np.linalg.norm(corner_1) > 0)
    assume(np.linalg.norm(corner_2) > 0)

    grid = utils.generate_uniform_grid(
        ncols=ncols,
        nrows=nrows,
        corner_1=corner_1,
        corner_2=corner_2,
        origin=origin,
    )
    assert grid.shape[0] == ncols * nrows


def test_generate_uniform_grid():
    grid = utils.generate_uniform_grid(
        ncols=3,
        nrows=3,
        corner_1=np.array([0, 1]),
        corner_2=np.array([1, 0]),
        origin=np.array([0, 0]),
    )
    np.testing.assert_array_equal(
        grid,
        np.array(
            [
                [0, 0],
                [0, 0.5],
                [0, 1],
                [0.5, 0],
                [0.5, 0.5],
                [0.5, 1],
                [1, 0],
                [1, 0.5],
                [1, 1],
            ]
        ),
        strict=True,
    )


def test_generate_uniform_grid_errors():
    with pytest.raises(ValueError):
        utils.generate_uniform_grid(
            ncols=1,
            nrows=10,
            corner_1=np.array([0, 1]),
            corner_2=np.array([1, 0]),
            origin=np.array([0, 0]),
        )
    with pytest.raises(ValueError):
        utils.generate_uniform_grid(
            ncols=10,
            nrows=1,
            corner_1=np.array([0, 1]),
            corner_2=np.array([1, 0]),
            origin=np.array([0, 0]),
        )
    with pytest.raises(ValueError):
        utils.generate_uniform_grid(
            ncols=10,
            nrows=10,
            corner_1=np.array([0, 0]),
            corner_2=np.array([1, 0]),
            origin=np.array([0, 0]),
        )
    with pytest.raises(ValueError):
        utils.generate_uniform_grid(
            ncols=10,
            nrows=10,
            corner_1=np.array([1, 0]),
            corner_2=np.array([0, 0]),
            origin=np.array([0, 0]),
        )
